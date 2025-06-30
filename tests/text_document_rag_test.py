from langchain_core.documents import Document
from intellitube.rag import TextDocumentRAG
from intellitube.utils import (
    YTContentData, 
    download_youtube_audio_or_transcript,
    webvtt_2_json, webvtt_2_langchain_documents
)
from typing import List


def get_transcript_as_documents() -> List[Document]:
    url = 'https://www.youtube.com/watch?v=W3I3kAg2J7w&t=231s'
    
    data: YTContentData = download_youtube_audio_or_transcript(
        video_url=url
    )
    print("Downloaded Data: ", data, end='\n\n')

    return webvtt_2_langchain_documents(vtt_file_path=data.transcript_path)


if __name__ == "__main__":
    documents = get_transcript_as_documents()

    tdr = TextDocumentRAG()
    tdr.add_documents(
        documents, split_text=True,
        split_config={
            "chunk_size": 1024,
            "chunk_overlap": 256
        }
    )

    docs = tdr.retriever.invoke("who am I?", k=5)
    print(docs)
