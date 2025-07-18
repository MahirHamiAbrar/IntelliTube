import operator
from langchain_core.documents import Document
from typing_extensions import Annotated, List, TypedDict

class SummarizerAgentState(TypedDict):
    """Overall State of the Agent"""
    documents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str

class SummarizerSummaryState(TypedDict):
    """Map node's state"""
    content: str
