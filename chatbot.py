import os
from loguru import logger
from typing import Any, List, Dict, Annotated, Sequence, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import (AIMessageChunk,
    BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.messages.utils import message_chunk_to_message

from langchain_core.documents import Document
from langchain.chat_models import init_chat_model

from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import StateGraph, START, END
from intellitube.vector_store import VectorStoreManager


# Chat messages will be stored here
class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


class DocumenData:
    document: Document
    summary: str


# initialize the chat model
temperature = 0.0
# response_llm = init_chat_model(
    # model="granite3.3:8b", model_provider="ollama", 
    # stream_usage=True, temperature=temperature
# )

response_llm = init_chat_model(
    model="gemini-2.0-flash", model_provider="google_genai", 
    stream_usage=True, temperature=temperature
)
# llm = response_llm.bind_tools()


documents: List[DocumenData] = []  # list of documents (raw document + summary)
document_rag = VectorStoreManager()


def add_document(document: Document) -> None:
    """Adds document to the vector store"""
    global documents
    
    document_data = DocumenData()
    document_data.document = document
    document_data.summary = ""
    # document_data.summary = summarize(response_llm, document)
    
    documents.append(document_data)
    
    document_rag.add_documents(
        [document_data.document], split_text=True,
        split_config={
            "chunk_size": 512,
            "chunk_overlap": 128
        },
        skip_if_collection_exists=True,
    )


# Create tools
retriever_tool = create_retriever_tool(
    document_rag.vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.6}
    ),
    # document_rag.retriever,
    "retrieve_documents",
    "Search and return documents from the vector storage database"
)

def generate_response(state: MessagesState) -> MessagesState:
    usr_msg: HumanMessage = state["messages"][-1].content
    print(f"Invoking with Query: {usr_msg}")

    retrieved_docs = retriever_tool.invoke(
        usr_msg, k=5
    )

    print(retrieved_docs, end="\n" + ("-"*40) + "\n\n\nAssistant: ")

    system_prompt = SystemMessage((
        "You are a very helpful assistant." 
        "Help the user by responding to their queries in a professional manner."
        "\nHere are some contexts that might/might not help you to answer the query."
        "\nIf the query is a general query, then answer it normally."
        "\nBut, if the query is neither a general query nor can be answered from the given context, just say - you can't help."
        "\nDO NOT make up answers on your own."
        f"\nContexts:{'\n\n'.join(doc.page_content if type(doc) == Document else doc for doc in retrieved_docs)}"
        # f"\nContexts:{'\n\n'.join(doc for doc in retrieved_docs)}"
    ))
    
    # We follow this style of passing system prompt so that it can have the most updated 
    # context in it. (Good practice for future version, for now it's not impactful at all)
    ai_msg: AIMessage = response_llm.invoke(
        [system_prompt] + state["messages"]
    )
    return {"messages": ai_msg}


# Build the graph
graph = (
    StateGraph(MessagesState)
     .add_node(generate_response)
     .add_edge(START, "generate_response")
     .add_edge("generate_response", END)
)

# Compile the graph to get the agent
agent = graph.compile()
agent.get_graph().draw_png("images/chatbot_graph.png", fontname="jetbrains mono")

# from intellitube.utils import mermaid2png
# mermaid2png.draw_png_dark(agent, "images/chatbot_graph_2.png")


# Invoke the graph
# response = agent.invoke({"messages": [HumanMessage("Hi, what's up?")]})
# print(response)

from langchain_community.document_loaders import WebBaseLoader
add_document(
    WebBaseLoader("https://lilianweng.github.io/posts/2024-07-07-hallucination/").load()[0]
)
# print(documents[0])
# print(documents[0].document)
# print(documents[0].summary)

def chat() -> None:

    # intialize the message object
    messages: List[BaseMessage] = []
    user_input = input(">> ")
    
    while user_input.strip().lower() != "quit":

        # add the human message to the message list
        messages.append(HumanMessage(user_input))
        # messages[-1].pretty_print()
        
        # create stream response
        ai_msg = agent.stream(
            {"messages": messages},  stream_mode="messages",
        )

        # all ai-response-chunks will be aggregated into this variable 
        full_message: AIMessageChunk = None

        # stream the chunks to the output
        print("Assistant: ", end="", flush=True)
        for chunk, metadata in ai_msg:
            full_message = chunk if full_message is None else full_message + chunk
            print(chunk.content, end="", flush=True)
        print("\n")
        
        # add the AI message to the message list
        messages.append(message_chunk_to_message(full_message))
        # messages[-1].pretty_print()
        
        user_input = input(">> ")


"""

This discussion centers on addressing "extrinsic hallucinations" in large language models (LLMs), where they produce incorrect or misleading information, particularly for complex topics needing external knowledge. Various evaluation benchmarks like TruthfulQA and FEVER are used to assess factual accuracy. Proposed solutions include:

- Retrieval-Augmented Generation (RAG) methods such as Self-RAG and Recite-LM that fetch relevant data before generating responses.
- Inference-time intervention techniques like ITI for truthful answers.
- Fine-tuning models on factual datasets, exemplified by FLAME and other fine-tuning approaches.
- Layer-wise verification methods such as DoLa to enhance factual consistency.
- Hallucination detection tools including SelfCheckGPT and zero-resource black-box detection.
- Selective prediction strategies allowing models to decline answering if uncertain, thus reducing misinformation spread.
- Human feedback integration in learning processes, as seen with WebGPT, for improved reliability and factual accuracy.

The post underscores continuous research efforts aimed at minimizing hallucinations in LLMs for more dependable and truthful AI systems.

"""