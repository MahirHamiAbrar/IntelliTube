from loguru import logger
from pathlib import Path
from typing_extensions import Literal, Union

from intellitube.utils import ChatManager
from intellitube.tools import document_loader_tools
from .states import (
    AgentState, QueryExtractorResponseState, RetrieverNodeState
)

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)

from langgraph.types import Command, Send


# initialize new chat
chatman = ChatManager.new_chat()
logger.debug(f"NEW CHAT INITIALIZED. ID: {chatman.chat_id}")


# NODE 01: QUERY EXTRACTOR NODE
def extract_query(
    user_message: HumanMessage, llm_custom: BaseChatModel = None
) -> QueryExtractorResponseState:
    structured_llm = (llm_custom or llm).with_structured_output(QueryExtractorResponseState)
    response = structured_llm.invoke([user_message])
    return response

def router_node(state: AgentState) -> AgentState:
    user_message: HumanMessage = state.messages[-1]
    return {"query_extractor_response": extract_query(user_message)}

def select_route(state: AgentState) -> Literal["load_document", "retrieve_documents"]:
    extractor_response = state.query_extractor_response
    if not extractor_response.url:
        return "retrieve_documents"
    return "load_document"


# NODE 02: DOCUMENT LOADER NODE
document_loader_functions = {
    "document": document_loader_tools.load_document,
    "youtube_video": document_loader_tools.load_youtube_transcript,
    "website": document_loader_tools.load_webpage
}

loaded_docs: dict[str, list[QueryExtractorResponseState, list[Document]]] = {}

def load_document_node(state: AgentState):
    data: QueryExtractorResponseState = state.query_extractor_response
    
    # 1. Already Loaded?
    #   1. YES? [GO TO RETRIEVER]
    #   2. NO? [GO TO STEP 2]
    # 2. Try to load it
    #   1. Error Loading: Return ToolMessage("Error") to Chat LLM
    #   2. Successful Loading: [GO TO SUMMARIZER LLM]

    # 1. ALREADY LOADED?
    if data.url in loaded_docs:
        # update the state and redirect to the retriever node
        return Send(
            node="retriever",
            arg=RetrieverNodeState(
                query=state.messages[-1], documents=loaded_docs[data.url][1],
                document_info=loaded_docs[data.url][0]
            )
        )

    # 2. Try to load it
    func = document_loader_functions.get(data.urlof, None)
    if not func:
        # state update + redirection to the chat_agent node (with err msg)
        return Command(
            update={"messages": [ToolMessage("Error Loading Document. Invalid Function Call from LLM!")]},
            goto="chat_agent"
        )

    docs: Union[Exception, Document] = func(data.url)
    # check for error
    if isinstance(docs, Exception):
        # state update + redirection to the chat_agent node (with err msg)
        return Command(
            update={"messages": [ToolMessage(f"Error Loading Document. Error Details: {str(docs)}")]}, 
            goto="chat_agent"
        )

    # loading successful; save document info
    loaded_docs[data.url] = [data, docs]
    # implement logic for redirection to summarizer llm with new state object
    return Command(
        update=RetrieverNodeState(
            query=state.messages[-1], documents=loaded_docs[data.url][1],
            document_info=loaded_docs[data.url][0]
        ),
        goto="summarizer"
    )
