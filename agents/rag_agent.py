import os
from typing_extensions import Optional

from intellitube.utils import ChatManager
from intellitube.vector_store import VectorStoreManager

from langchain_core.language_models import BaseChatModel

# self query
# metadata filtering

class RAGAgent:
    _llm: BaseChatModel
    _chat_manager: ChatManager
    _vector_store: VectorStoreManager

    @property
    def llm(self) -> BaseChatModel:
        return self._llm
    
    @property
    def chat_manager(self) -> ChatManager:
        return self._chat_manager
    
    @property
    def vectorstore(self) -> VectorStoreManager:
        return self._vector_store

    def __init__(self,
        llm: BaseChatModel,
        chat_manager: ChatManager,
        vector_store: VectorStoreManager
    ) -> None:
        self._llm = llm
        self._chat_manager = chat_manager
        self._vector_store = vector_store
