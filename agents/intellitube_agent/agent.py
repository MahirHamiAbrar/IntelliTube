from loguru import logger
from pathlib import Path
from typing_extensions import Literal, Union

from intellitube.llm import init_llm
from intellitube.utils import ChatManager
from intellitube.tools import document_loader_tools
from intellitube.vector_store import VectorStoreManager
from intellitube.utils.path_manager import intellitube_dir
from intellitube.agents.summarizer_agent import SummarizerAgent
from .states import (
    AgentState, DocumentData, MultiQueryData,
    QueryExtractorResponseState, RetrieverNodeState
)
from .prompts import chat_agent_prompt, multi_query_prompt

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)

from langgraph.types import Command, Send
from langgraph.graph import StateGraph, START, END

# initialize new LLM
llm = init_llm(model_provider='google')

# initialize new chat
chatman = ChatManager.new_chat()
logger.debug(f"NEW CHAT INITIALIZED. ID: {chatman.chat_id}")


# initialize vector store
vdb = VectorStoreManager(
    path_on_disk = chatman.chat_dirpath,
    collection_path_on_disk = Path(chatman.chat_dirpath) / "collection",
    collection_name = chatman.chat_id,
)


# NODE 01: QUERY EXTRACTOR NODE

# helper function
def extract_query(
    user_message: HumanMessage, llm_custom: BaseChatModel = None
) -> QueryExtractorResponseState:
    structured_llm = (llm_custom or llm).with_structured_output(QueryExtractorResponseState)
    response = structured_llm.invoke([user_message])
    return response

# actual node
def router_node(state: AgentState) -> AgentState:
    user_message: HumanMessage = state.messages[-1]
    return {"query_extractor_response": extract_query(user_message)}

# select route
def select_route(state: AgentState) -> Literal["load_document", "retrieve_documents"]:
    if not state.query_extractor_response.url:
        return "retrieve_documents"
    return "load_document"


# NODE 02: DOCUMENT LOADER NODE
document_loader_functions = {
    "document": document_loader_tools.load_document,
    "youtube_video": document_loader_tools.load_youtube_transcript,
    "website": document_loader_tools.load_webpage
}

loaded_docs: dict[str, DocumentData] = {}

def load_document_node(
    state: AgentState
) -> Union[Send, Command[AgentState | RetrieverNodeState]]:
    """
    # Load a document from the given URL/Local Path.

    How it works:
         1. Already Loaded?
           1. YES? [GO TO RETRIEVER]
           2. NO? [GO TO STEP 2]
         2. Try to load it
           1. Error Loading: Return ToolMessage("Error") to Chat LLM
           2. Successful Loading: [GO TO SUMMARIZER LLM]
    """
    data: QueryExtractorResponseState = state.query_extractor_response

    # 1. ALREADY LOADED?
    if data.url in loaded_docs:
        # update the state and redirect to the retriever node
        return Send(
            node="retriever",
            arg=RetrieverNodeState(query=state.messages[-1], data=loaded_docs[data.url])
        )

    # 2. Try to load it
    func = document_loader_functions.get(data.urlof, None)
    if not func:
        # state update + redirection to the chat_agent node (with err msg)
        return Command(
            goto="chat_agent",
            update={"messages": [ToolMessage("Error Loading Document. Invalid Function Call from LLM!")]},
        )

    docs: Union[Exception, Document] = func(data.url)
    # check for error
    if isinstance(docs, Exception):
        # state update + redirection to the chat_agent node (with err msg)
        return Command(
            goto="chat_agent",
            update={"messages": [ToolMessage(f"Error Loading Document. Error Details: {str(docs)}")]}, 
        )

    # loading successful; save document info
    loaded_docs[data.url] = DocumentData(documents=docs, metadata=data)
    logger.info("Adding document to vector database ...")
    # save documents in vector store
    vdb.add_documents(
        docs, split_text=True,
        split_config={
            "chunk_size": 512,
            "chunk_overlap": 128
        },
        skip_if_collection_exists=True,
    )
    # implement logic for redirection to summarizer llm with new state object
    return Command(
        goto="summarizer",
        update=RetrieverNodeState(query=state.messages[-1], data=loaded_docs[data.url]),
    )


# NODE 03: Summarizer Node
summarizer = SummarizerAgent(llm=llm)
def summarizer_node(state: RetrieverNodeState) -> Send:
    summary = summarizer.summarize(documents=state.documents)
    state.data["summary"] = summary
    return Send(node="retriever", arg=state)

# NODE 04: Generate Multi Query & rewrite Query
def multiquery_gen_node(state: RetrieverNodeState) -> RetrieverNodeState:
    messages = ChatPromptTemplate.from_messages(
        [multi_query_prompt, state.query]
    )
    structured_llm = llm.with_structured_output(MultiQueryData)
    query_data = structured_llm.invoke(messages.format_messages(summary=state.data["summary"]))
    state.query_data = query_data
    return state

# NODE 05: Retrieve Information from database
retriever = vdb.vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.6}
)
def retriever_node(state: RetrieverNodeState) -> RetrieverNodeState:
    docs = retriever.invoke(state.query, k=3)
    docs.append(retriever.invoke(state.query_data.rewritten_query, k=3))
    state.retrieved_docs = docs
    return state

# NODE 06: Chat Agent Node
def chat_agent_node(state: RetrieverNodeState) -> AgentState:
    messages = ChatPromptTemplate.from_messages(
        [chat_agent_prompt, state.query]
    )
    docs = "\n\n".join(
        f"Document {i + 1}:\n" + doc.page_content 
        for i, doc in enumerate(state.retrieved_docs)
    )
    ai_msg: AIMessage = llm.invoke(messages.format_messages(docs=docs))
    return {"messages": [ai_msg]}


# build the graph
graph = (
    StateGraph(AgentState)

    # add nodes
    .add_node("router", router_node)
    .add_node("document_loader", load_document_node)
    .add_node("summarizer", summarizer_node)
    .add_node("multiquery_generator", multiquery_gen_node)
    .add_node("retriever", retriever_node)
    .add_node("chat_agent", chat_agent_node)
    
    # add_edges
    .add_edge(START, "router")
    .add_conditional_edges(
        "router",
        select_route,
        {
            "load_document": "document_loader",
            "retrieve_documents": "retriever",
        }
    )
    .add_edge("retriever", "chat_agent")
    .add_edge("chat_agent", END)
)

agent = graph.compile()


def chat_loop() -> None:
    print(f"Chat ID: {chatman.chat_id}")
    usr_msg: str = input(">> ").strip()

    while usr_msg.lower() != "/exit":
        usr_msg = HumanMessage(usr_msg)
        chatman.add_message(usr_msg)
        state = AgentState(messages=chatman.chat_messages)
        chatman.chat_messages = agent.invoke(state)["messages"]
        ai_msg: AIMessage = chatman.chat_messages[-1]
        ai_msg.pretty_print()
        usr_msg: str = input(">> ").strip()
    
    chatman.save_chat()
    chatman.remove_unlisted_chats()

def test_agent():
    save_path = Path(intellitube_dir) / "images/intellitube_ai_agent.png"
    save_path.write_bytes(agent.get_graph().draw_mermaid_png())
