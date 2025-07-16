from langgraph.graph import StateGraph, START, END
from langchain_core.language_models import BaseChatModel

from intellitube.agents.base_agent import BaseAgent


class ChatAgent(BaseAgent):

    def __init__(self, llm: BaseChatModel) -> None:
        BaseAgent.__init__(self, llm)
    
    def build_graph(self) -> StateGraph:
        pass
