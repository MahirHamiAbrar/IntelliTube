"""
WORKFLOW EXPLAINED:

Example User Message:
    How to implement an agentic rag system according to this website?
    https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/


Steps to process this user message:
    0. [START]
    1. Separate the Query & URL
    2. Try to load the URL
        1. Already loaded?
            1. [GO TO: RETRIEVER STEP (3)]
        2. Not loaded?
            1. Load it
            2. Generate a Summary
                1. Save the summary
            3. Save the document
            4. [GO TO: RETRIEVER STEP (3)]
    3. Retrieve Information form database
    4. Pass information to Chat Agent
        1. Generate a response
        2. Show it to the user
        3. [END]

"""

from typing_extensions import (
    Annotated, Sequence,
    Dict, Literal, TypedDict, Optional, Union
)

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.language_models import BaseChatModel

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from intellitube.llm import init_llm
from intellitube.tools import document_loader_tools
from intellitube.prompts import router_agent_prompts


class DocumentInfoModel(TypedDict):
    document: Document
    summary: str

llm = init_llm(model_provider='google')
document_database: Dict[str, DocumentInfoModel] = {}


def add_document(document: Document) -> None:
    key = document.metadata["source"]
    document_info = DocumentInfoModel(
        document=document,
        summary=""
    )
    document_database[key] = document_info

def document_already_loaded(key: Union[Document, str]) -> bool:
    if isinstance(key, Document):
        key = key.metadata["source"]
    return document_database.get(key) is not None


# =============================================
# STEP 01: Separate the Query & URL
# =============================================

# define Router Agent Output Schema
class QueryExtractorAgentResponse(BaseModel):
    instruction: str = Field(description=(
        "The user's instruction quoted word-for-word with any URLs or Paths removed.\n"
        "Preserve the casing, punctuation, and wording. Do NOT fix typos or grammar."
    ))
    analysis: str = Field(description=(
        "Aanalyze the RAW data and describe what the user actually meant in one sentence."
        "Be clear and concise about the user's intention."
    ))
    url: Optional[str] = Field(default=None, description=(
        "The URL or local path provided by the user, if any."
        "Should be extracted separately from the user-query."
        "If there is no URL or file path, leave this as null (do not fabricate one).\n"
        "Example: 'https://example.com/page', 'C:/Documents/myfile.txt', './notes.md'"
    ))
    urlof: Optional[
        Literal["youtube_video", "website", "document"]
    ] = Field(default=None, description=(
        "The type of content the `url` field refers to:\n"
        "- 'youtube_video': if it's a YouTube video link\n"
        "- 'website': for general websites or web pages\n"
        "- 'document': for file paths (like .txt, .pdf, .md, etc.)\n"
        "If no URL/path is provided, this should be null."
    ))



class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    """Conversation messages"""
    query_extractor_response: QueryExtractorAgentResponse = None
    """Router Agent Response Status"""


def extract_query(
    user_message: HumanMessage, llm_custom: BaseChatModel = None
) -> QueryExtractorAgentResponse:
    structured_llm = (llm_custom or llm).with_structured_output(QueryExtractorAgentResponse)
    response = structured_llm.invoke([user_message])
    return response


def router_node(state: AgentState) -> AgentState:
    user_message: HumanMessage = state.messages[-1]
    # state.query_extractor_response = extract_query(user_message)
    return {"query_extractor_response": extract_query(user_message)}

def select_route(state: AgentState) -> Literal["load_document", "retrieve_documents"]:
    extractor_response = state.query_extractor_response
    if not extractor_response.url:
        return "retrieve_documents"
    return "load_document"

def print_route(state: AgentState, route: str) -> AgentState:
    print("Selected Route: ", route)
    return state


# =============================================
# BUILD THE GRAPH
# =============================================
graph = (
    StateGraph(state_schema=AgentState)
    # add nodes
    .add_node(router_node)
    .add_node("printer_node1", lambda state: print_route(state, "load_document"))
    .add_node("printer_node2", lambda state: print_route(state, "retrieve_documents"))

    # add edges
    .add_edge(START, "router_node")
    .add_conditional_edges(
        "router_node",
        select_route,
        {
            "load_document": "printer_node1",
            "retrieve_documents": "printer_node2",
        }
    )
    .add_edge("printer_node1", END)
    .add_edge("printer_node2", END)
)

agent = graph.compile()



if __name__ == "__main__":
    user_message = (
        "I wrote the idea somewhere in draft.txt... just quote it pls"
        # "Do you know what an idea is?"
    )

    resp = agent.invoke({"messages": [user_message]})
    print(resp)

    # response = extract_query(user_message)
    # print(f"{user_message = }", end='\n\n')
    # print(response)
