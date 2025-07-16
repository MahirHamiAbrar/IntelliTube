from langgraph.graph import StateGraph, START, END
from intellitube.agents.base_agent import BaseAgent


class ChatAgent(BaseAgent):

    def __init__(self) -> None:
        BaseAgent.__init__(self)
    
    def build_graph(self) -> StateGraph:
        pass
