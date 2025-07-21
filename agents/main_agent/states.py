from pydantic import BaseModel, Field
from typing_extensions import (
    Annotated, List, Literal, Sequence, TypedDict, Optional
)
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

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
