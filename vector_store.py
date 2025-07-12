# from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore

import os
from loguru import logger
from typing import Any, List, Dict, Optional, Union

from pydantic import BaseModel
from langchain_core.embeddings import Embeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.chat_models import BaseChatModel


class VectorStoreManager:
    # embedding_model_name: str = "bge-m3:567m"
    embedding_model_name: str = 'sentence-transformers/all-MiniLM-L12-v2'
    # embedding_model_name: str = 'models/gemini-embedding-exp-03-07'
    # embedding_model_name: str = 'jinaai/jina-embeddings-v3'
    path_on_disk: str = "test_data/qdrant_vector_store"
    collection_path_on_disk: str = "test_data/qdrant_vector_store/collection"
    collection_name: str = "text-document-rag"
    collection_exists: bool = False

    @property
    def vectorstore(self) -> QdrantVectorStore:
        return self._vector_store
    
    @property
    def client(self) -> QdrantClient:
        return self._client

    def __init__(self,
        embedding_model: Optional[Union[BaseModel, Embeddings]] = None,
        path_on_disk: Optional[str] = None,
        collection_path_on_disk: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self.path_on_disk = path_on_disk or self.path_on_disk
        self.collection_path_on_disk = collection_path_on_disk or self.collection_path_on_disk
        self.collection_name = collection_name or self.collection_name
        
        self.load_embedding_model(embedding_model)
        self.init_vector_store()

    def load_embedding_model(self, embedding_model: Optional[Union[BaseModel, Embeddings]] = None) -> None:
        # self._embeddings = OllamaEmbeddings(model=self.embedding_model_name)
        if 'gemini' in self.embedding_model_name:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self._embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model_name)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = (
                embedding_model if embedding_model
                else HuggingFaceEmbeddings(model=self.embedding_model_name)
            )
            
    
    def init_vector_store(self) -> None:
        if os.path.exists(
            os.path.join(self.collection_path_on_disk, self.collection_name)
        ):
            logger.debug("Loading vector store from existing collection")
            self._vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.collection_name,
                embedding=self._embeddings,
                path=self.path_on_disk
            )
            self.collection_exists = True
        else:
            logger.debug("Creaing Client...")
            self._client = QdrantClient(path=self.path_on_disk)
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=len(self._embeddings.embed_query("hehe")), distance=Distance.COSINE
                )
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
        skip_if_collection_exists: bool = True,
    ) -> None:
        
        if skip_if_collection_exists and self.collection_exists:
            logger.warning(
                "Collection exists, so NOT ADDING documents." \
                "To override this behaviour, set `skip_if_collection_exists` parameter to `False`.")
            return
        
        if split_text:
            logger.info("Splitting text...")
            self._text_splitter = RecursiveCharacterTextSplitter(**split_config)
            documents = self._text_splitter.split_documents(documents)
        
        logger.info("Adding documents...")
        self._vector_store.add_documents(documents)
    
    def generate_answer(self, 
        query: str, llm: BaseChatModel,
        retrieved_docs: List[Document]
    ) -> None:
        
        document_context = "\n\n".join(
            f"Document #{i + 1}\n" + document.page_content
            for i, document in enumerate(retrieved_docs)
        )

# Relevant Questions: {matched_questions}
        template = PromptTemplate.from_template(
            template="""Using the provided context and associated questions, give a clear, engaging, and direct answer to the user's question in a conversational tone. Speak directly to the user, use active voice, and focus on the key information from the context that best matches the query. If the context or questions don't fully address the query, briefly acknowledge this and provide the most relevant information available.

Context: {context}

Question: {query}

Answer:
"""
        )

        chain = (
            {"query": lambda x: x["query"], "context": lambda _: document_context}
            | template
            | llm
        )

        return chain.invoke({ "query": query })

