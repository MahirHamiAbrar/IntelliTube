import asyncio
from typing import (
    Any, Callable, Dict, List,
    Literal, Optional, Tuple, 
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnableSerializable

from langchain.chains.combine_documents.reduce import (
    acollapse_docs, split_list_of_docs
)

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph

from intellitube.agents.base_agent import BaseAgent
from .prompts import map_prompt, reduce_prompt
from .states import SummarizerAgentState, SummarizerSummaryState


class SummarizerAgent(BaseAgent):
    max_tokens: int
    _map_chain: RunnableSerializable = None
    _reduce_chain: RunnableSerializable = None

    @property
    def map_chain(self) -> RunnableSerializable:
        if not self._map_chain:
            self._map_chain = (
                ChatPromptTemplate([map_prompt]) | self.llm | StrOutputParser()
            )
        return self._map_chain
    
    @property
    def reduce_chain(self) -> RunnableSerializable:
        if not self._reduce_chain:
            self._reduce_chain = (
                ChatPromptTemplate([reduce_prompt]) | self.llm | StrOutputParser()
            )
        return self._reduce_chain

    def __init__(self,
        llm: BaseChatModel,
        max_tokens: int = 2048
    ) -> None:
        BaseAgent.__init__(self, llm)
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
            .add_node("generate_summary", self.generate_summary)
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
        super().build_graph()
        return graph
    
    async def asummarize(self,
        documents: List[Document],
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> SummarizerAgentState:
        """Asynchronous method to summarize the given list of documents together."""
        state = SummarizerAgentState(documents=documents)
        state = await self.agent.ainvoke(
            input=state,
            config=config or {"recursion_limit": 30},
            **kwargs
        )
        return state
    
    def summarize(self,
        documents: List[Document],
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> SummarizerAgentState:
        """Synchronous method to summarize the given list of documents together."""
        return asyncio.run(self.asummarize(
            documents, config, **kwargs
        ))
    
    async def stream_asummarize(self,
        documents: List[Document],
        config: Optional[RunnableConfig] = None,
        stream_updater_callback: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], SummarizerAgentState]:
        """Asynchronous method to summarize the given list of documents together."""
        steps = []
        state = SummarizerAgentState(documents=documents)
        
        async for step in self.agent.astream(
            input=state,
            config=config or {"recursion_limit": 30},
            **kwargs
        ):
            if callable(stream_updater_callback):
                stream_updater_callback(step)
            
            results = {
                k: v
                for _dict in list(step.values())
                for k, v in _dict.items()
            }
            state = SummarizerAgentState(**state, **results)
            steps.append(step)
            
        return steps, state
    
    def stream_summarize(self,
        documents: List[Document],
        config: Optional[RunnableConfig] = None,
        stream_updater_callback: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], SummarizerAgentState]:
        """Synchronous method to summarize the given list of documents together."""
        return asyncio.run(self.stream_asummarize(
            documents, config, stream_updater_callback, **kwargs
        ))
