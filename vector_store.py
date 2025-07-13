from pathlib import Path
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    # fallback embedding model
    _fallback_embedding_model_name: str = 'sentence-transformers/all-MiniLM-L12-v2'
    _embedding_model: Embeddings = None
    
    # define paths
    _db_local_path: Path = Path.cwd() / "test_data/qdrant_vector_store"
    _collection_local_path: Path = _db_local_path / "collection"

    # collection names & status
    _collection_name: str = "my collection"
    _collection_exists: bool = False

    @property
    def fallback_embedding_model_name(self) -> str:
        return self._embedding_model_name
    
    @property
    def embedding_model(self) -> Embeddings:
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
        return self._collection_exists
    
    def __init__(self) -> None:
        pass

    def load_fallback_embedding_model(self) -> None:
        from langchain_huggingface import HuggingFaceEmbeddings
        self._embedding_model = HuggingFaceEmbeddings(
            model=self.fallback_embedding_model_name
        )

