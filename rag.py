# from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

import os
from loguru import logger
from typing import Any, List, Dict


class TextDocumentRAG:
    # embedding_model_name: str = "bge-m3:567m"
    embedding_model_name: str = 'sentence-transformers/all-MiniLM-L12-v2'
    path_on_disk: str = "test_data/qdrant_vector_store"
    collection_path_on_disk: str = "test_data/qdrant_vector_store/collection"
    collection_name: str = "text-document-rag"

    def __init__(self) -> None:
        # self._embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        self._embeddings = HuggingFaceEmbeddings(model=self.embedding_model_name)

        if os.path.exists(
            os.path.join(self.collection_path_on_disk, self.collection_name)
        ):
            logger.debug("Loading vector store from existing collection")
            self._vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.collection_name,
                embedding=self._embeddings,
                path=self.path_on_disk
            )
        else:
            logger.debug("Creaing Client...")
            self._client = QdrantClient(path=self.path_on_disk)
            self._client.create_collection(
                collection_name=self.collection_name,
                # vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.debug("Creaing vector store")
            self._vector_store = QdrantVectorStore(
                client=self._client,
                collection_name=self.collection_name,
                embedding=self._embeddings
            )

        self.retriever = self._vector_store.as_retriever()

    
    def add_documents(self, 
        documents: List[Document], 
        split_text: bool = False, 
        split_config: Dict[str, Any] = {},
    ) -> None:
        
        if split_text:
            logger.info("Splitting text...")
            self._text_splitter = RecursiveCharacterTextSplitter(**split_config)
            documents = self._text_splitter.split_documents(documents)
        
        logger.info("Adding documents...")
        self._vector_store.add_documents(documents)
