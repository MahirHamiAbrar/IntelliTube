import operator
from typing import (
    Annotated, Dict, List, Literal, Optional, TypedDict
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable
from langchain_core.output_parsers import StrOutputParser

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph

from intellitube.prompts.summarizer_agent_prompts import (
    map_prompt, reduce_prompt
)


class AgentState(TypedDict):
    """Overall State of the Agent"""
    documents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


class SummaryState(TypedDict):
    """Map node's state"""
    content: str


class SummarizerAgent:
    max_tokens: int
    llm: BaseChatModel

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

    def __init__(self,
        llm: BaseChatModel,
        max_tokens: int = 2048
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
    
    def length_function(self, documents: List[Document]) -> int:
        """Get number of tokens for input contents."""
        return sum(self.llm.get_num_tokens(doc.page_content) for doc in documents)
    
    async def generate_summary(self, state: SummaryState) -> Dict[str, List[str]]:
        """Generate summary of a document"""
        llm_response = await self.map_chain.ainvoke(state['documents'])
        return {"summaries": [llm_response]}
    
    def map_summaries(self, state: AgentState) -> List[Send]:
        """Maps the summary of the given documents"""
        return [
            Send("generate_summary", {"content": content})
            for content in state["documents"]
        ]
    
    def collect_summaries(self, state: AgentState) -> Dict[str, List[Document]]:
        """Collects the summaries of the mapped-documents"""
        return {
            "collapsed_summaries": [
                Document(summary)
                for summary in state["summaries"]
            ]
        }



if __name__ == '__main__':
    from intellitube.llm import init_llm
    llm = init_llm('groq')
    agent = SummarizerAgent(llm)
    out = agent.map_chain.invoke({"context": "This is a report about why we breath. We breath because we breath. It is impossible to live without breathing. So, breathing is essential for out life."})
    print(out)
