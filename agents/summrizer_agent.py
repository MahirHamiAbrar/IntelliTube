import operator
from typing import (
    Any, Annotated, Dict, List, Literal, Optional, TypedDict
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser

from langchain.chains.combine_documents.reduce import (
    acollapse_docs, split_list_of_docs
)

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from intellitube.prompts.summarizer_agent_prompts import (
    map_prompt, reduce_prompt
)


class SummarizerAgentState(TypedDict):
    """Overall State of the Agent"""
    documents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


class SummarizerSummaryState(TypedDict):
    """Map node's state"""
    content: str


class SummarizerAgent:
    max_tokens: int
    llm: BaseChatModel

    _graph: StateGraph = None
    _agent: CompiledStateGraph = None
    _map_chain: RunnableSerializable = None
    _reduce_chain: RunnableSerializable = None

    @property
    def map_chain(self) -> RunnableSerializable:
        if not self._map_chain:
            self._map_chain = ChatPromptTemplate([map_prompt]) | self.llm | StrOutputParser()
        return self._map_chain
    
    @property
    def reduce_chain(self) -> RunnableSerializable:
        if not self._reduce_chain:
            self._reduce_chain = ChatPromptTemplate([reduce_prompt]) | self.llm | StrOutputParser()
        return self._reduce_chain
    
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

    def __init__(self,
        llm: BaseChatModel,
        max_tokens: int = 2048
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
    
    def length_function(self, documents: List[Document]) -> int:
        """Get number of tokens for input contents."""
        return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)
    
    async def generate_summary(self, state: SummarizerSummaryState) -> Dict[str, List[str]]:
        """Generate summary of a document"""
        llm_response = await self.map_chain.ainvoke(state['content'])
        return {"summaries": [llm_response]}
    
    def map_summaries(self, state: SummarizerAgentState) -> List[Send]:
        """Maps the summary of the given documents"""
        return [
            Send("generate_summary", {"content": content})
            for content in state["documents"]
        ]
    
    def collect_summaries(self, state: SummarizerAgentState) -> Dict[str, List[Document]]:
        """Collects the summaries of the mapped-documents"""
        return {
            "collapsed_summaries": [
                Document(summary)
                for summary in state["summaries"]
            ]
        }
    
    async def collapse_summaries(self, state: SummarizerAgentState) -> Dict[str, list]:
        """Collapse the collected summaries"""
        doc_lists = split_list_of_docs(
            state["collapsed_summaries"], self.length_function, self.max_tokens
        )
        results = []
        for doc_list in doc_lists:
            results.append(
                await acollapse_docs(doc_list, self.reduce_chain.invoke)
            )
        return {"collapsed_summaries": results}
    
    def should_collapse(self,
        state: SummarizerAgentState
    ) -> Literal["collapse_summaries", "generate_final_summary"]:
        n_tokens = self.length_function(state["collapsed_summaries"])
        return (
            "collapse_summaries"
            if n_tokens > self.max_tokens
            else "generate_final_summary"
        )
    
    async def generate_final_summary(self, state: SummarizerAgentState) -> Dict[str, Any]:
        llm_response = await self.reduce_chain.ainvoke(state["collapsed_summaries"])
        return {"final_summary": llm_response}
    
    def build_graph(self) -> StateGraph:
        graph = (
            StateGraph(SummarizerAgentState)
            # add nodes
            .add_node("generate_summary", self.generate_summary)  # same as before
            .add_node("collect_summaries", self.collect_summaries)
            .add_node("collapse_summaries", self.collapse_summaries)
            .add_node("generate_final_summary", self.generate_final_summary)

            # add edges
            .add_conditional_edges(START, self.map_summaries, ["generate_summary"])
            .add_edge("generate_summary", "collect_summaries")
            .add_conditional_edges("collect_summaries", self.should_collapse)
            .add_conditional_edges("collapse_summaries", self.should_collapse)
            .add_edge("generate_final_summary", END)
        )
        self._agent = None  # reset the agent variable
        return graph
