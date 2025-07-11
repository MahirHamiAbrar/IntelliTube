import os
from loguru import logger
from typing import List

from langchain_core.documents import Document

from intellitube.agents import init_llm
from intellitube.utils import ChatManager
from intellitube.rag import TextDocumentRAG
from intellitube.tools import document_loader_tools


# initialize chat manager (new chat)
chat_manager = ChatManager.new_chat()
logger.debug(f"Chat ID: {chat_manager.chat_id}")

# initialize an LLM
llm = init_llm(model_provider='google')


# initialize rag system
document_rag = TextDocumentRAG(
    path_on_disk=chat_manager.chat_dirpath,
    collection_path_on_disk=os.path.join(chat_manager.chat_dirpath, "collection"),
    collection_name=chat_manager.chat_id,
)

def add_to_vdb(docuemnts: List[Document]) -> None:
    # convert to a list of document(s) if not already!
    if type(docuemnts) == Document:
        docuemnts = [docuemnts]
    
    document_rag.add_documents(
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
