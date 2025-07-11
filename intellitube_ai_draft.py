import os
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import (
    Annotated, Sequence, List, Literal, TypedDict, Optional
)

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage, SystemMessage, HumanMessage, ToolMessage, BaseMessage
)

from langgraph.graph.message import add_messages

from intellitube.llm import init_llm
from intellitube.utils import ChatManager
from intellitube.rag import TextDocumentRAG
from intellitube.tools import document_loader_tools
from intellitube.prompts import router_agent_prompts


# initialize chat manager (new chat)
chat_manager = ChatManager.new_chat()
logger.debug(f"Chat ID: {chat_manager.chat_id}")

# initialize an LLM
llm = init_llm(model_provider='google')


# initialize rag system
document_rag = TextDocumentRAG(
    path_on_disk=chat_manager.chat_dirpath,
    collection_path_on_disk=os.path.join(chat_manager.chat_dirpath, "collection"),
    collection_name=chat_manager.chat_id,
)

def add_to_vdb(docuemnts: List[Document]) -> None:
    # convert to a list of document(s) if not already!
    if type(docuemnts) == Document:
        docuemnts = [docuemnts]
    
    document_rag.add_documents(
        docuemnts, split_text=True,
        split_config={
            "chunk_size": 512,
            "chunk_overlap": 128
        },
        skip_if_collection_exists=True,
    )

# document loader functions
document_loader_functions = {
    "document": document_loader_tools.load_document,
    "youtube_video": document_loader_tools.load_youtube_transcript,
    "website": document_loader_tools.load_webpage
}

# define Router Agent Output Schema
class RouterAgentResponse(BaseModel):
    user_query: str = Field(description=(
        "The user's original query EXACTLY as it appears, without any modification, rewording, or interpretation.\n"
        "You MUST NOT include any URLs, file paths, or hyperlinks in this field — only the natural language query.\n"
        "Preserve the casing, punctuation, and wording. Do NOT fix typos or grammar."
    ))
    url: Optional[str] = Field(default=None, description=(
        "The exact URL or local document/file path mentioned in the user's input.\n"
        "If there is no URL or file path, leave this as null (do not fabricate one).\n"
        "Example: 'https://example.com/page', 'C:/Documents/myfile.txt', './notes.md'"
    ))
    url_of: Optional[Literal["youtube_video", "website", "document"]] = Field(default=None, description=(
        "The type of content the `url` field refers to:\n"
        "- 'youtube_video': if it's a YouTube video link\n"
        "- 'website': for general websites or web pages\n"
        "- 'document': for file paths (like .txt, .pdf, .md, etc.)\n"
        "If no URL/path is provided, this should be null."
    ))


# define Chat Agent Output Schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    router_response: Optional[RouterAgentResponse] = None
    retrieved_docs: Optional[List[Document]] = None


# Router Agent Nodes
def router_agent_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(RouterAgentResponse)
    agent_resp: RouterAgentResponse = structured_llm.invoke(
        [router_agent_prompts.system_prompt, state["messages"][-1]]
    ) 
    return {"messages": [HumanMessage(agent_resp.user_query)], "router_response": agent_resp}

# query router node
def query_router_node(state: AgentState) -> Literal["use_loader", "use_retriever"]:
    return "use_retriever" if not state["router_response"].url_of else "use_loader"

# document loader node
def document_loader_node(state: AgentState) -> Literal["success", "fail"]:
    logger.info(f'{state["router_response"] = }')
    loader_func = document_loader_functions.get(state["router_response"].url_of)
    logger.info(f"{loader_func = }")
    status = loader_func(state["router_response"].url)
    return "success" if status else "fail"

# retriever node
RETRIEVER = document_rag.vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.6}
)

def document_retriever_node(state: AgentState) -> AgentState:
    state["retrieved_docs"] = RETRIEVER.invoke(state["router_response"].user_query)
    print("\n\n\n")
    print(state["retrieved_docs"], end='\n\n')
    return state
