from loguru import logger
from typing_extensions import Any, Dict, List, Literal, Union

from intellitube.llm import init_llm
from intellitube.utils import ChatManager
from intellitube.tools import document_loader_tools
from intellitube.vector_store import VectorStoreManager
from intellitube.utils.path_manager import intellitube_dir
from intellitube.agents.summarizer_agent import SummarizerAgent
from .states import SummarizerAgentState, QueryExtractorData
from .prompts import chat_agent_prompt, multi_query_prompt

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
INIT = True

llm: BaseChatModel = None
chatman: ChatManager = None
vdb: VectorStoreManager = None

if INIT:
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

def extract_and_route_query(
    state: MessagesState
) -> Literal["load_document", "retriever"]:
    """Take the user query, fetch the URL from it and then redirect it 
    either to the document loader node or to the retriever node."""
    logger.debug("Running router node")

    structured_llm = llm.with_structured_output(QueryExtractorData)
    data: QueryExtractorData = structured_llm.invoke([state['messages'][-1]])
    print(data)

    if data["url"]:
        return Send("load_document", {**data, **state})
    return Send("retriever", data)

# ERROR: {'urlof': 'website', 'analysis': 'The user is asking a general question and has not provided enough information to determine their intent. They need to provide more context.', 'instruction': 'why?', 'url': 'null'}

# ------------- NODE 02: DOCUMENT LOADER NODE -------------
loaded_docs: dict[str, QueryExtractorData] = {}

document_loader_functions = {
    "document": document_loader_tools.load_document,
    "youtube_video": document_loader_tools.load_youtube_transcript,
    "website": document_loader_tools.load_webpage
}

def load_document_node(
    data: QueryExtractorData
) -> Union[Send, Command[Dict[str, Any]]]:
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

    logger.debug("Running load_document_node\n\n")

    # 1. ALREADY LOADED?
    if data["url"] in loaded_docs:
        # update the state and redirect to the retriever node
        return Send("retrieve_documents", data)

    # 2. Try to load it
    func = document_loader_functions.get(data["urlof"], None)
    if not func:
        # state update + redirection to the chat_agent node (with err msg)
        return Command(
            goto="generate_answer",
            update={"messages": [
                ToolMessage("Error Loading Document. Invalid Function Call from LLM!", tool_call_id='')
            ]},
        )

    docs: Union[Exception, Document] = func(data["url"])
    # check for error
    if isinstance(docs, Exception):
        # state update + redirection to the chat_agent node (with err msg)
        return Command(
            goto="generate_answer",
            update={"messages": [ToolMessage(f"Error Loading Document. Error Details: {str(docs)}", tool_call_id='')]}, 
        )

    # loading successful; save document info
    # save everything except the messages, it'll besaved elsewhere
    loaded_docs[data["url"]] = {k: data[k] for k in data.keys() - {'messages'}}
    logger.info("Adding document to vector database ...")
    # save documents in vector store
    vdb.add_documents(
        docs, split_text=True,
        split_config={
            "chunk_size": 512,
            "chunk_overlap": 128
        },
        skip_if_collection_exists=False,
    )
    # implement logic for redirection to summarizer llm with new state object
    return Send("summarizer", {**data, "documents": docs})


# ------------- NODE 03: SUMMARIZER NODE -------------
summarizer: SummarizerAgent = None
def summarizer_node(state: SummarizerAgentState) -> Send:
    logger.debug("Running summarizer_node\n\n")
    global summarizer
    if not summarizer:
        summarizer = SummarizerAgent(llm=llm)

    summary = summarizer.summarize(documents=state["documents"])
    state["summary"] = summary
    return Send("retriever", state)


# ------------- NODE 04: RETRIEVE INFORMATION FROM DATABASE -------------
retriever = vdb.vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.6}
)

def retriever_node(
    data: QueryExtractorData
) -> Dict[str, Any]:
    logger.debug("Running retriever_node\n\n")
    docs = retriever.invoke(data["instruction"], k=3)
    # docs.append(retriever.invoke(state.query_data.rewritten_query, k=3))

    docs_prompt = "\n\n".join(
        (
            f"Document {i + 1}: {doc.page_content}\n"
            f"Document {i + 1} metadata: "
            ' '.join(f'{k}: {v};' for k, v in doc.metadata.items())
        )
        for i, doc in enumerate(docs)
    )

    print("DOCS PROMPT:", docs_prompt)
    # return {**data, "retrieved_documents": docs}
    return {"messages": [ToolMessage(content=docs_prompt, tool_call_id="")]}


# ------------- NODE 05: CHAT AGENT NODE -------------
# NODE 05: Chat Agent Node
def chat_agent_node(state: MessagesState) -> MessagesState:
    logger.debug("Running chat_agent_node\n\n")
    logger.success(state)

    messages = ChatPromptTemplate.from_messages(
        [chat_agent_prompt, *state["messages"]]
    )
    # docs = "\n\n".join(
    #     f"Document {i + 1}:\n" + doc.page_content 
    #     for i, doc in enumerate(state.retrieved_docs)
    # )
    ai_msg: AIMessage = llm.invoke(messages.format_messages())
    return {"messages": [ai_msg]}



# ======================== GRAPH ========================
graph = (
    StateGraph(MessagesState)
    
    # add nodes
    .add_node("query_router", lambda state: state)  # just for show in the png graph
    .add_node("load_document", lambda state: state) # pass through
    .add_node("summarizer", summarizer_node)
    .add_node("chat_agent", chat_agent_node)
    .add_node("retriever", retriever_node)

    # add edges
    .add_edge(START, "query_router")
    .add_conditional_edges(
        "query_router", extract_and_route_query, ["load_document", "retriever"]
    )
    .add_conditional_edges(
        "load_document", load_document_node, {
            "generate_answer": "chat_agent",
            "summarize_content": "summarizer",
            "retrieve_documents": "retriever"
        }
    )
    .add_edge("summarizer", "retriever")
    .add_edge("retriever", "chat_agent")
    .add_edge("chat_agent", END)
)

agent = graph.compile()

save_path = intellitube_dir / "images/intellitube_ai_graph.png"
save_path.write_bytes(agent.get_graph().draw_mermaid_png())


def test_intellitube_ai() -> None:
    resp = agent.invoke({
        "messages": [HumanMessage("What is this website about? https://wikipedia.org")]
    })
    print(resp)

def chat_loop() -> None:
    print(f"Chat ID: {chatman.chat_id}")
    usr_msg: str = input(">> ").strip()

    while usr_msg.lower() != "/exit":
        usr_msg = HumanMessage(usr_msg)
        chatman.add_message(usr_msg)
        
        state = MessagesState(messages=chatman.chat_messages)
        chatman.chat_messages = agent.invoke(state)["messages"]

        # for update, messages in agent.stream(input=state, stream_mode=['updates', 'messages']):
        #     # print(updates.keys())
        #     # logger.success(update.keys())
        #     logger.success(update)
        #     # print(update)
        #     logger.info(messages)
        
        ai_msg: AIMessage = chatman.chat_messages[-1]
        ai_msg.pretty_print()
        usr_msg: str = input(">> ").strip()
    
    chatman.save_chat()
    chatman.remove_unlisted_chats()
