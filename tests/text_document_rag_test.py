from pprint import pprint
from langchain_core.documents import Document
from intellitube.rag import TextDocumentRAG
from intellitube.utils import (
    YTContentData, 
    download_youtube_audio_or_transcript,
    webvtt_2_json, webvtt_2_langchain_documents, webvtt_2_str
)
from typing import List

from langchain.chat_models import init_chat_model


def get_transcript_as_documents() -> List[Document]:
    url = 'https://www.youtube.com/watch?v=W3I3kAg2J7w&t=231s'
    
    data: YTContentData = download_youtube_audio_or_transcript(
        video_url=url
    )
    print("Downloaded Data: ", data, end='\n\n')

    # return webvtt_2_langchain_documents(vtt_file_path=data.transcript_path)

    return Document(
        page_content=webvtt_2_str(vtt_file_path=data.transcript_path),
        metadata={
            "source": url
        }
    )


if __name__ == "__main__":
    documents = get_transcript_as_documents()
    # llm = init_chat_model(model="llama3.2:3b", model_provider="ollama", temperature=0)
    # llm = init_chat_model(model="granite3.3:8b", model_provider="ollama", temperature=0)
    llm = init_chat_model(model="mistral-nemo", model_provider="ollama", temperature=0)
    
    # pprint(documents.page_content)
    # exit(0)

    tdr = TextDocumentRAG()
    tdr.add_documents(
        [documents], split_text=True,
        split_config={
            "chunk_size": 512,
            "chunk_overlap": 128
        },
        skip_if_collection_exists=True,
    )

    query = "Who am I?"
    # query = "Who is the author?"
    # query = "What is a weird job?"
    # query = "What is not a good advice?"
    # query = "What is a bad advice?"
    
    tdr.retriever = tdr._vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.6}
    )

    docs = tdr.vectorstore.similarity_search_with_relevance_scores(
        query=query,
        k = 5,
    )

    # docs = tdr.retriever.invoke("Why was President Nempahrd thanked?", k=5)
    # docs = tdr.retriever.invoke(query, k=5)
    pprint(docs)

    # answer = tdr.generate_answer(query, llm, docs)
    # print(f"\n\n{answer = }")
    # print(f"\n\n{answer.content}")

