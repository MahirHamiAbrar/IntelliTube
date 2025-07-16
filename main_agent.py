from typing_extensions import Union
from langchain_core.documents import Document

from intellitube.tools import document_loader_tools


# class DocumentMetadata(TypedDict):
    # document_type: 

document_database = {}

def add_document(document: Document) -> None:
    document_database[document.metadata["source"]] = document

def document_already_loaded(data: Union[Document, str]) -> bool:
    if isinstance(data, Document):
        data = data.metadata["source"]
    return document_database.get(data) is not None

