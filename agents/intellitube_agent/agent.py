from loguru import logger
from pathlib import Path
from typing_extensions import Literal, Union

from intellitube.utils import ChatManager
from intellitube.tools import document_loader_tools
from intellitube.vector_store import VectorStoreManager
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


# initialize vector store
vdb = VectorStoreManager(
    path_on_disk = chatman.chat_dirpath,
    collection_path_on_disk = Path(chatman.chat_dirpath) / "collection",
    collection_name = chatman.chat_id,
)


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
    if not state.query_extractor_response.url:
        return "retrieve_documents"
    return "load_document"


# NODE 02: DOCUMENT LOADER NODE
document_loader_functions = {
    "document": document_loader_tools.load_document,
    "youtube_video": document_loader_tools.load_youtube_transcript,
    "website": document_loader_tools.load_webpage
}

loaded_docs: dict[str, list[QueryExtractorResponseState, list[Document]]] = {}

def load_document_node(
    state: AgentState
) -> Union[Send[RetrieverNodeState], Command[AgentState, RetrieverNodeState]]:
    """
    # Load a document from the given URL/Local Path.

    How it works:
         1. Already Loaded?
           1. YES? [GO TO RETRIEVER]
           2. NO? [GO TO STEP 2]
         2. Try to load it
           1. Error Loading: Return ToolMessage("Error") to Chat LLM
           2. Successful Loading: [GO TO SUMMARIZER LLM]
    """
    data: QueryExtractorResponseState = state.query_extractor_response

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
            goto="chat_agent",
            update={"messages": [ToolMessage("Error Loading Document. Invalid Function Call from LLM!")]},
        )

    docs: Union[Exception, Document] = func(data.url)
    # check for error
    if isinstance(docs, Exception):
        # state update + redirection to the chat_agent node (with err msg)
        return Command(
            goto="chat_agent",
            update={"messages": [ToolMessage(f"Error Loading Document. Error Details: {str(docs)}")]}, 
        )

    # loading successful; save document info
    loaded_docs[data.url] = [data, docs]

    # save documents in vector store
    vdb.add_documents(
        docs, split_text=True,
        split_config={
            "chunk_size": 512,
            "chunk_overlap": 128
        },
        skip_if_collection_exists=True,
    )
    # implement logic for redirection to summarizer llm with new state object
    return Command(
        goto="summarizer",
        update=RetrieverNodeState(
            query=state.messages[-1], documents=loaded_docs[data.url][1],
            document_info=loaded_docs[data.url][0]
        ),
    )


# NODE 03: Summarizer Node
def summarizer_node(state: RetrieverNodeState):
    return Send(node="retriever", arg={})

# NODE 04: Retrieve Information from database
def retriever_node(state: RetrieverNodeState):
    return Send(node="chat_agent", arg={})
