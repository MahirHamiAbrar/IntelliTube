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
from intellitube.rag import TextDocumentRAG


# Chat messages will be stored here
class MessagesState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# initialize the chat model
temperature = 0.0
response_llm = init_chat_model(
    model="granite3.3:8b", model_provider="ollama", 
            stream_usage=True, temperature=temperature)
# llm = response_llm.bind_tools()


documents: List[Document] = []  # list of documents
document_rag = TextDocumentRAG()


def add_document(document: Document) -> None:
    """Adds document to the vector store"""
    global documents
    documents.append(document)
    document_rag.add_documents(
        [document], split_text=True,
        split_config={
            "chunk_size": 512,
            "chunk_overlap": 128
        },
        skip_if_collection_exists=True,
    )


# Create tools
retriever_tool = create_retriever_tool(
    document_rag.retriever,
    "retrieve_documents",
    "Search and return documents from the vector storage database"
)

def generate_response(state: MessagesState) -> MessagesState:
    system_prompt = SystemMessage("You are a very helpful assistant. Help the user by responding to their queries in a professional manner.")
    
    # We follow this style of passing system prompt so that it can have the
    # updated context in it. (good practice for future version, for now it's useless)
    ai_msg: AIMessage = response_llm.invoke(
        [system_prompt] + state["messages"]
    )
    return {"messages": ai_msg}


# Build the graph
graph = StateGraph(MessagesState)

graph.add_node(generate_response)

graph.add_edge(START, "generate_response")
graph.add_edge("generate_response", END)

# Compile the graph to get the agent
agent = graph.compile()
agent.get_graph().draw_png("images/chatbot_graph.png", fontname="jetbrains mono")


# Invoke the graph
# response = agent.invoke({"messages": [HumanMessage("Hi, what's up?")]})
# print(response)


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

        # all ai-response-chunks will be aggregrated into this variable 
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
