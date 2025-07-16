from pathlib import Path
from typing import Union
from abc import ABC, abstractmethod

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.language_models import BaseChatModel


class BaseAgent(ABC):
    llm: BaseChatModel
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

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
    
    def save_graph_image(self, path: Union[Path, str]) -> None:
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError(f'Provided "path" is neither a Path() instance nor a str() instance.')
        
        mermaid_png = self.agent.get_graph().draw_mermaid_png()
        path.write_bytes(mermaid_png)
