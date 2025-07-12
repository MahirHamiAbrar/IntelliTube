import os
from typing_extensions import Optional

from intellitube.llm import init_llm
from intellitube.utils import ChatManager
from intellitube.rag import TextDocumentRAG

from langchain_core.language_models import BaseChatModel

# self query
# metadata filtering

class RAGAgent:
    _llm: BaseChatModel
    _chat_manager: ChatManager
    _document_rag: TextDocumentRAG

    @property
    def llm(self) -> BaseChatModel:
        return self._llm
    
    @property
    def chat_manager(self) -> ChatManager:
        return self._chat_manager
    
    @property
    def document_rag(self) -> TextDocumentRAG:
        return self._document_rag

    def __init__(self,
        llm: BaseChatModel,
        chat_manager: ChatManager,
        document_rag: TextDocumentRAG
    ) -> None:
        self._llm = llm
        self._chat_manager = chat_manager
        self._document_rag = document_rag
