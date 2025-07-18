from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, List, Sequence, TypedDict


class ChatAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    """Chat conversation messages with the agent"""
    documents: List[Document] = []
    """Documents added by the user"""
