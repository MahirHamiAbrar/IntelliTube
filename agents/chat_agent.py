from typing_extensions import Annotated, List, Sequence, TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel

from intellitube.utils import ChatManager
from intellitube.agents.base_agent import BaseAgent


class ChatAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    """Chat conversation messages with the agent"""
    documents: List[Document] = []
    """Documents added by the user"""


class ChatAgent(BaseAgent):
    chat_manager: ChatManager

    def __init__(self,
        llm: BaseChatModel,
        chat_manager: ChatManager,
    ) -> None:
        BaseAgent.__init__(self, llm)
        self.chat_manager = chat_manager
    
    def build_graph(self) -> StateGraph:
        graph = StateGraph(state_schema=ChatAgentState)
        super().build_graph()
        return graph
    
    def youtube_video_loader_node(self, state: ChatAgentState) -> ChatAgentState:
        pass
    
    def summarizer_node(self, document: Document) -> ChatAgentState:
        pass
