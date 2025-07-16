from typing_extensions import Annotated, Sequence, TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel

from intellitube.utils import ChatManager
from intellitube.agents.base_agent import BaseAgent


class ChatAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    """Chat conversation messages with the agent"""


class ChatAgent(BaseAgent):

    def __init__(self,
        llm: BaseChatModel,
        chat_manager: ChatManager,
    ) -> None:
        BaseAgent.__init__(self, llm)
    
    def build_graph(self) -> StateGraph:
        graph = StateGraph(state_schema=ChatAgentState)
        super().build_graph()
        return graph
