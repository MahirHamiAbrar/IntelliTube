from pydantic import BaseModel, Field
from typing_extensions import (
    Annotated, List, Literal, Optional, Sequence, TypedDict
)

from langchain_core.documents import Documents
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class QueryExtractorResponseState(BaseModel):
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
    query_extractor_response: QueryExtractorResponseState = None
    """Router Agent Response Status"""

class DocumentData(TypedDict):
    documents: List[Documents]
    """Retrieved documents"""
    metadata: QueryExtractorResponseState
    """Document information"""
    summary: Optional[str] = None

class RetrieverNodeState(BaseModel):
    query: str
    """The user query for retrieval context"""
    data: DocumentData
    """Document information"""
    multi_query: List[str] = None
    """Generated multiple query for multi-query retrieval"""
    rewritten_query: str = None
    """Improved contextual rewritten form of the original user query"""
