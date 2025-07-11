import os
from loguru import logger
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader, WebBaseLoader
)

from intellitube.utils import (
    YTContentData,
    webvtt_2_str, download_youtube_audio_or_transcript,
)


def load_youtube_transcript(youtube_url: str) -> Union[Exception, List[Document]]:
    """Load the given YouTube video's transcript to the vector database.
    It is required to answer user-queries based on the the Transcript context."""
    document: Document
    try:
        logger.debug("Loading Youtube Transcript...")
        
        # download the youtube transcript (or audio if transcript not available)
        yt_video_data: YTContentData = download_youtube_audio_or_transcript(
            video_url=youtube_url,
        )

        # convert the WEBVTT format trancript to a plain text string
        vtt_str = webvtt_2_str(vtt_file_path=yt_video_data.transcript_path)
        logger.debug(vtt_str[:100])    # print first 100 characters
        document = Document(vtt_str)
    except Exception as e:
        logger.error(str(e))
        return e
    return [document]


def load_document(document_path: str) -> Union[Exception, List[Document]]:
    """Load the given Document's content to the vector database.
    It is required to answer user-queries based on the the Document context."""
    documents: List[Document]
    try:
        logger.debug("Loading Document...")
        ext = os.path.splitext(document_path)[1][1:].lower()

        if ext == 'pdf':
            documents = PyPDFLoader(document_path).load()
        elif ext in ['txt', 'py']:
            with open(document_path, 'r') as file:
                documents = [Document(
                    page_content=file.read(),
                    metadata={ "source": document_path }
                )]
        else:
            return Exception(f"Invalid file type: {ext}")
    
    except Exception as e:
        logger.error(str(e))
        return e
    return documents


def load_webpage(webpage_url: str) -> Union[Exception, List[Document]]:
    """Load the given WebSite's content to the vector database.
    It is required to answer user-queries based on the the WebPage's context."""
    documents: List[Document]
    try:
        documents = WebBaseLoader(webpage_url).load()
        logger.debug("Loading Webpage...")
    except Exception as e:
        logger.error(str(e))
        return e
    return documents
