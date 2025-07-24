from loguru import logger
from typing_extensions import List, Literal, Union

from intellitube.utils import ChatManager
from intellitube.agents.base_agent import BaseAgent
from intellitube.tools import document_loader_tools
from intellitube.vector_store import VectorStoreManager
from .states import AgentState, RouterAgentResponse
from .prompts import router_agent_system_prompt
from intellitube.agents.chat_agent.prompts import (
    system_prompt as chat_agent_system_prompt
)

from langgraph.graph import START, END, StateGraph

from langchain_core.messages import (
    AIMessage, HumanMessage, ToolMessage
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStoreRetriever


class IntelliTubeAI(BaseAgent):
    _chat_manager: ChatManager
    _vdb: VectorStoreManager
    _retriever: VectorStoreRetriever = None
    _similarity_score_threshold: float = 0.6

    document_loader_functions = {
        "document": document_loader_tools.load_document,
        "youtube_video": document_loader_tools.load_youtube_transcript,
        "website": document_loader_tools.load_webpage
    }

    @property
    def chat_manager(self) -> ChatManager:
        return self._chat_manager
    
    @property
    def vdb(self) -> VectorStoreManager:
        return self._vdb
    
    @property
    def retriever(self) -> VectorStoreRetriever:
        if not self._retriever:
            self._retriever = self.vdb.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': self._similarity_score_threshold}
            )
        return self._retriever

    def __init__(self, 
        llm: BaseChatModel,
        chat_manager: ChatManager,
        vector_store_manager: VectorStoreManager,
    ) -> None:
        BaseAgent.__init__(self, llm=llm)
        
        self._chat_manager = chat_manager
        self._vdb = vector_store_manager
    
    def add_to_vdb(self, documents: List[Document]) -> None:
        # convert to a list of document(s) if not already!
        if type(documents) == Document:
            documents = [documents]
        
        self.vdb.add_documents(
            documents, split_text=True,
            split_config={
                "chunk_size": 512,
                "chunk_overlap": 128
            },
            skip_if_collection_exists=False,
        )
    
    # ========== DEFINE NODES ==========

    def router_agent_node(self, state: AgentState) -> AgentState:
        """Router Agent Nodes"""
        structured_llm = self.llm.with_structured_output(RouterAgentResponse)
        messages = ChatPromptTemplate.from_messages(
            [router_agent_system_prompt, state["messages"][-1]]
        )
        agent_resp: RouterAgentResponse = structured_llm.invoke(
            messages.format_messages()
        )
        # return {"messages": [HumanMessage(agent_resp.user_query)], "router_response": agent_resp}
        return {"router_response": agent_resp}

    def query_router_node(self, state: AgentState) -> Literal["use_loader", "use_retriever"]:
        """query router node"""
        return "use_retriever" if not state["router_response"].url_of else "use_loader"

    def document_loader_node(self, state: AgentState) -> Literal["success", "fail"]:
        """document loader node"""
        logger.info(f'{state["router_response"] = }')
        loader_func = self.document_loader_functions.get(state["router_response"].url_of)
        logger.info(f"{loader_func = }")
        documents: Union[Exception, List[Document]] = loader_func(state["router_response"].url)
        if type(documents) == Exception:
            return "fail"
        self.add_to_vdb(documents)
        return "success"
    
    def document_retriever_node(self, state: AgentState) -> AgentState:
        """document retriever node"""
        print(f'{state["router_response"].user_query = }')
        state["retrieved_docs"] = self.retriever.invoke(state["router_response"].user_query)
        print("\n\n\n")
        print(state["retrieved_docs"], end='\n\n')
        return state
    
    def chat_agent_node(self, state: AgentState) -> AgentState:
        """A Chat Agent Node!"""
        docs = (
            "\n\n".join(f"Source #{i + 1}: {document.page_content}" for i, document in enumerate(state["retrieved_docs"]))
            if state.get("retrieved_docs") else ""
        )
        context = '\n' + docs if docs else '[No Context Available.]'
        context_source = f" from {state['router_response'].url} {state['router_response'].url_of}"
        
        messages = ChatPromptTemplate.from_messages(
            [chat_agent_system_prompt, *state["messages"]]
        )
        ai_msg: AIMessage = self.llm.invoke(messages.format_messages(
            context=context, context_source=context_source
        ))
        # return None to reset every other variable except "messages"
        return {"messages": [ai_msg], "retrieved_docs": None, "router_response": None}
    
    def deliver_failed_message_node(self, state: AgentState) -> AgentState:
        return {
            "messages": [ToolMessage(
                content=f"failed to load {state['router_response'].url}", 
                tool_call_id=self.chat_manager.chat_id
                )
            ]
        }
    
    def build_graph(self) -> StateGraph:
        graph = (
            StateGraph(state_schema=AgentState)
            .add_node("router_agent", self.router_agent_node)
            .add_node("chat_agent", self.chat_agent_node)
            .add_node("document_loader", lambda state: state)
            .add_node("document_retriever", self.document_retriever_node)
            .add_node(
                "deliver_failed_message",
                self.deliver_failed_message_node
            )
            
            .add_edge(START, "router_agent")
            .add_conditional_edges(
                source="router_agent",
                path=self.query_router_node,
                path_map={
                    "use_loader": "document_loader",
                    "use_retriever": "document_retriever",
                }
            )
            .add_conditional_edges(
                source="document_loader",
                path=self.document_loader_node,
                path_map={
                    "fail": "deliver_failed_message",
                    "success": "document_retriever",
                }
            )
            .add_edge("deliver_failed_message", "chat_agent")
            .add_edge("document_retriever", "chat_agent")
            .add_edge("chat_agent", END)
        )
        super().build_graph()
        return graph
    
    def cli_chat_loop(self) -> None:
        print(f"Chat ID: {self.chat_manager.chat_id}")
        usr_msg: str = input(">> ").strip()

        while usr_msg.lower() != "/exit":
            usr_msg = HumanMessage(usr_msg)
            
            self.chat_manager.add_message(usr_msg)
            self.chat_manager.chat_messages = self.agent.invoke(
                {"messages": self.chat_manager.chat_messages}
            )["messages"]
            
            # for update in agent.stream({"messages": chat.chat_messages}, stream_mode="updates"):
                # print(update)
            ai_msg: AIMessage = self.chat_manager.chat_messages[-1]
            ai_msg.pretty_print()
            usr_msg: str = input(">> ").strip()
        
        self.chat_manager.save_chat()
        self.chat_manager.remove_unlisted_chats()
