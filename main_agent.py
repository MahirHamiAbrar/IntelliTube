from typing_extensions import (
    Dict, Literal, TypedDict, Optional, Union
)

from pydantic import BaseModel, Field
from langchain_core.documents import Document

from intellitube.llm import init_llm
from intellitube.tools import document_loader_tools


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

# Router Agent Nodes
def router_agent_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(RouterAgentResponse)
    messages = ChatPromptTemplate.from_messages(
        [router_agent_prompts.system_prompt, state["messages"][-1]]
    )
    agent_resp: RouterAgentResponse = structured_llm.invoke(
        messages.format_messages()
    )
    return {"messages": [HumanMessage(agent_resp.user_query)], "router_response": agent_resp}


