from abc import ABC, abstractmethod
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph


class BaseAgent(ABC):
    _graph: StateGraph = None
    _agent: CompiledStateGraph = None

    @property
    def graph(self) -> StateGraph:
        if not self._graph:
            self._graph = self.build_graph()
        return self._graph
    
    @property
    def agent(self) -> CompiledStateGraph:
        if not self._agent:
            self._agent = self.graph.compile()
        return self._agent
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        pass
