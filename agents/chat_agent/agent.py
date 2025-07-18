from langgraph.graph import StateGraph

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel

from .states import ChatAgentState
from intellitube.utils import ChatManager
from intellitube.agents.base_agent import BaseAgent



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
