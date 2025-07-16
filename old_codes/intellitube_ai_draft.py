import os
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import (
    Annotated, List, Literal, Sequence, TypedDict, Union, Optional
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)

from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph

from intellitube.llm import init_llm
from intellitube.utils import ChatManager
from intellitube.vector_store import VectorStoreManager
from intellitube.tools import document_loader_tools
from intellitube.prompts import (
    chat_agent_prompts, router_agent_prompts
)


# initialize chat manager (new chat)
chat_manager = ChatManager.new_chat()
logger.debug(f"Chat ID: {chat_manager.chat_id}")

# initialize an LLM
llm = init_llm(model_provider='google')


# initialize rag system
vectorstore = VectorStoreManager(
    path_on_disk=chat_manager.chat_dirpath,
    collection_path_on_disk=os.path.join(chat_manager.chat_dirpath, "collection"),
    collection_name=chat_manager.chat_id,
)

def add_to_vdb(docuemnts: List[Document]) -> None:
    # convert to a list of document(s) if not already!
    if type(docuemnts) == Document:
        docuemnts = [docuemnts]
    
    vectorstore.add_documents(
        docuemnts, split_text=True,
        split_config={
            "chunk_size": 512,
            "chunk_overlap": 128
        },
        skip_if_collection_exists=True,
    )

# document loader functions
document_loader_functions = {
    "document": document_loader_tools.load_document,
    "youtube_video": document_loader_tools.load_youtube_transcript,
    "website": document_loader_tools.load_webpage
}

# define Router Agent Output Schema
class RouterAgentResponse(BaseModel):
    user_query: str = Field(description=(
        "The user's original query EXACTLY as it appears, without any modification, rewording, or interpretation.\n"
        "You MUST NOT include any URLs, file paths, or hyperlinks in this field — only the natural language query.\n"
        "Preserve the casing, punctuation, and wording. Do NOT fix typos or grammar."
    ))
    url: Optional[str] = Field(default=None, description=(
        "The exact URL or local document/file path mentioned in the user's input.\n"
        "If there is no URL or file path, leave this as null (do not fabricate one).\n"
        "Example: 'https://example.com/page', 'C:/Documents/myfile.txt', './notes.md'"
    ))
    url_of: Optional[Literal["youtube_video", "website", "document"]] = Field(default=None, description=(
        "The type of content the `url` field refers to:\n"
        "- 'youtube_video': if it's a YouTube video link\n"
        "- 'website': for general websites or web pages\n"
        "- 'document': for file paths (like .txt, .pdf, .md, etc.)\n"
        "If no URL/path is provided, this should be null."
    ))


# define Chat Agent Output Schema
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    router_response: Optional[RouterAgentResponse] = None
    retrieved_docs: Optional[List[Document]] = None


# Router Agent Nodes
def router_agent_node(state: AgentState) -> AgentState:
    structured_llm = llm.with_structured_output(RouterAgentResponse)
    messages = ChatPromptTemplate.from_messages(
        [router_agent_prompts.system_prompt, state["messages"][-1]]
    )
    agent_resp: RouterAgentResponse = structured_llm.invoke(
        messages.format_messages()
    )
    return {"messages": [HumanMessage(agent_resp.user_query)], "router_response": agent_resp}

# query router node
def query_router_node(state: AgentState) -> Literal["use_loader", "use_retriever"]:
    return "use_retriever" if not state["router_response"].url_of else "use_loader"

# document loader node
def document_loader_node(state: AgentState) -> Literal["success", "fail"]:
    logger.info(f'{state["router_response"] = }')
    loader_func = document_loader_functions.get(state["router_response"].url_of)
    logger.info(f"{loader_func = }")
    documents: Union[Exception, List[Document]] = loader_func(state["router_response"].url)
    if type(documents) == Exception:
        return "fail"
    add_to_vdb(documents)
    return "success"

# retriever node
RETRIEVER = vectorstore.vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.6}
)

# document retriever node
def document_retriever_node(state: AgentState) -> AgentState:
    state["retrieved_docs"] = RETRIEVER.invoke(state["router_response"].user_query)
    print("\n\n\n")
    print(state["retrieved_docs"], end='\n\n')
    return state

# Chat Agent Node
def chat_agent_node(state: AgentState) -> AgentState:
    """A Chat Agent"""
    docs = (
        "\n\n".join(f"Source #{i + 1}: {document.page_content}" for i, document in enumerate(state["retrieved_docs"]))
        if state.get("retrieved_docs") else ""
    )
    context = '\n' + docs if docs else '[No Context Available.]'
    context_source = f" from {state['router_response'].url} {state['router_response'].url_of}"
    
    messages = ChatPromptTemplate.from_messages(
        [chat_agent_prompts.system_prompt, *state["messages"]]
    )
    ai_msg: AIMessage = llm.invoke(messages.format_messages(
        context=context, context_source=context_source
    ))
    # return None to reset every other variable except "messages"
    return {"messages": [ai_msg], "retrieved_docs": None, "router_response": None}


# Create The AI Agent (Graph)
def deliver_failed_message_node(state: AgentState) -> AgentState:
    return {"messages": [ToolMessage(content=f"failed to load {state['router_response'].url}", tool_call_id=chat_manager.chat_id)]}

graph = (
    StateGraph(state_schema=AgentState)
    .add_node("router_agent", router_agent_node)
    .add_node("chat_agent", chat_agent_node)
    .add_node("document_loader", lambda state: state)
    .add_node("document_retriever", document_retriever_node)
    .add_node(
        "deliver_failed_message",
        deliver_failed_message_node
    )
    
    .add_edge(START, "router_agent")
    .add_conditional_edges(
        source="router_agent",
        path=query_router_node,
        path_map={
            "use_loader": "document_loader",
            "use_retriever": "document_retriever",
        }
    )
    .add_conditional_edges(
        source="document_loader",
        path=document_loader_node,
        path_map={
            "fail": "deliver_failed_message",
            "success": "document_retriever",
        }
    )
    .add_edge("deliver_failed_message", "chat_agent")
    .add_edge("document_retriever", "chat_agent")
    .add_edge("chat_agent", END)
)

agent = graph.compile()
agent.get_graph().draw_png(
    output_file_path=os.path.join(chat_manager.chat_dirpath, "agent_graph.png")
)

# the chat function
def chat_loop() -> None:
    print(f"Chat ID: {chat_manager.chat_id}")
    usr_msg: str = input(">> ").strip()

    while usr_msg.lower() != "/exit":
        usr_msg = HumanMessage(usr_msg)
        chat_manager.add_message(usr_msg)
        chat_manager.chat_messages = agent.invoke({"messages": chat_manager.chat_messages})["messages"]
        # for update in agent.stream({"messages": chat.chat_messages}, stream_mode="updates"):
            # print(update)
        ai_msg: AIMessage = chat_manager.chat_messages[-1]
        ai_msg.pretty_print()
        usr_msg: str = input(">> ").strip()
    chat_manager.save_chat()
    chat_manager.remove_unlisted_chats()


if __name__ == '__main__':
    chat_loop()
