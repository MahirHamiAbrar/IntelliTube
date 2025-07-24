from loguru import logger
from typing_extensions import Literal, Union

from intellitube.llm import init_llm
from intellitube.utils import ChatManager
from intellitube.tools import document_loader_tools
from intellitube.vector_store import VectorStoreManager
from intellitube.utils.path_manager import intellitube_dir
from intellitube.agents.summarizer_agent import SummarizerAgent
from .states import QueryExtractorState
# from .prompts import chat_agent_prompt, multi_query_prompt

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)

from langgraph.types import Command, Send
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState

# ======================== INIT ========================
# initialize new LLM
llm = init_llm(model_provider='google', model_name='gemini-2.5-flash')

# initialize new chat
chatman = ChatManager.new_chat()
logger.debug(f"NEW CHAT INITIALIZED. ID: {chatman.chat_id}")

# initialize vector store
vdb = VectorStoreManager(
    path_on_disk = chatman.chat_dirpath,
    collection_path_on_disk = chatman.chat_dirpath / "collection",
    collection_name = chatman.chat_id,
)


# ======================== INIT ========================

# ------------- NODE 01: QUERY EXTRACTOR NODE -------------

def extract_and_route_query(state: MessagesState) -> MessagesState:
    structured_llm = llm.with_structured_output(QueryExtractorState)
    data = structured_llm.invoke([state['messages'][-1]])
    print(data)
    return state


# ======================== GRAPH ========================
graph = (
    StateGraph(MessagesState)
    
    # add nodes
    .add_node("query_router", extract_and_route_query)

    # add edges
    .add_edge(START, "query_router")
    .add_edge("query_router", END)
)

agent = graph.compile()

save_path = intellitube_dir / "images/intellitube_ai_graph.png"
save_path.write_bytes(agent.get_graph().draw_mermaid_png())


def test_intellitube_ai() -> None:
    resp = agent.invoke({
        "messages": [HumanMessage("What is this website about? https://wikipedia.org")]
    })
    print(resp)
