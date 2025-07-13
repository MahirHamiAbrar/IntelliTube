from pathlib import Path
from loguru import logger
from typing_extensions import Optional, Union

from pydantic import BaseModel
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from qdrant_client.http import models
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore


class VectorStoreManager:
    # fallback embedding model
    _fallback_embedding_model_name: str = 'sentence-transformers/all-MiniLM-L12-v2'
    _embedding_model: Union[BaseModel, Embeddings] = None
    
    # define paths
    _db_local_path: Path = Path.cwd() / "test_data/qdrant_vector_store"
    _collection_local_path: Path = _db_local_path / "collection"

    # collection names & status
    _collection_name: str = "my collection"

    # vector store
    _client: QdrantClient = None
    _vector_store: QdrantVectorStore = None

    # retriever
    _retriever: VectorStoreRetriever = None

    @property
    def fallback_embedding_model_name(self) -> str:
        return self._fallback_embedding_model_name
    
    @property
    def embedding_model(self) -> Union[BaseModel, Embeddings]:
        return self._embedding_model
    
    @property
    def db_local_path(self) -> Path:
        return self._db_local_path
    
    @property
    def collection_local_path(self) -> Path:
        return self._collection_local_path
    
    @property
    def collection_name(self) -> str:
        return self._collection_name
    
    @collection_name.setter
    def collection_name(self, name: str) -> None:
        self._collection_name = name
    
    @property
    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)
    
    @property
    def client(self) -> QdrantClient:
        assert self._client is not None, "Initialize vector store first!"
        return self._client
    
    @property
    def vectorstore(self) -> QdrantVectorStore:
        assert self._vector_store is not None, "Initialize vector store first!"
        return self._vector_store
    
    @property
    def retriever(self) -> VectorStoreRetriever:
        """Summons the default retriever!"""
        if not self._retriever:
            self._retriever = self.vectorstore.as_retriever()
        return self._retriever
    
    def __init__(self,
        embedding_model: Optional[Union[BaseModel, Embeddings]] = None,
        path_on_disk: Optional[str] = None,
        collection_path_on_disk: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_config: Optional[models.VectorParams] = None,
        auto_init_vector_store: bool = True
    ) -> None:
        self._db_local_path = path_on_disk or self.db_local_path
        self._collection_local_path = collection_path_on_disk or self.collection_local_path
        self.collection_name = collection_name or self.collection_name
        self._embedding_model = embedding_model or self.load_fallback_embedding_model()

        if auto_init_vector_store:
            self.init_vector_store(vector_config=vector_config)
        
    def load_fallback_embedding_model(self) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model=self.fallback_embedding_model_name
        )
    
    def init_vector_store(self, vector_config: Optional[models.VectorParams] = None) -> None:
        if self.collection_local_path.exists():
            self._vector_store = QdrantVectorStore.from_existing_collection(
                collection_name=self.collection_name,
                embedding=self.embedding_model,
                path=self.db_local_path
            )
            logger.debug("Vector Store loaded from existing collection.")
        else:
            self._client = QdrantClient(path=self.db_local_path)
            # create the collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_config or models.VectorParams(
                    size=len(self.embedding_model.embed_query("hehe")),
                    distance=models.Distance.COSINE
                )
            )
            logger.debug("New Qdrant Client Initialized.")
            self._vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embedding_model
            )
            logger.debug("New Vector Store Created.")
