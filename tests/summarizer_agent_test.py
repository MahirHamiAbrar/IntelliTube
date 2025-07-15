from intellitube.llm import init_llm
from intellitube.agents import SummarizerAgent, SummarizerAgentState
from intellitube.utils import (
    YTContentData, download_youtube_audio_or_transcript, webvtt_2_str
)

from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import CharacterTextSplitter


def load_document() -> List[str]:
    # load a youtube video
    video_url = "https://www.youtube.com/watch?v=UF8uR6Z6KLc"
    video_data: YTContentData = download_youtube_audio_or_transcript(video_url)
    vtt_str = webvtt_2_str(vtt_file_path=video_data.transcript_path)

    # initialize text splitter
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )

    # split the texts
    return text_splitter.split_text(vtt_str)


def test_summarizer_agent(llm: BaseChatModel, docs: List[Document]) -> None:
    summarizer = SummarizerAgent(llm=llm)
    # summarizer.save_graph_image("images/summarizer_agent_graph.png")
    data: SummarizerAgentState = summarizer.summarize(docs)
    print(data["final_summary"])

def test_summarizer_agent_stream(llm: BaseChatModel, docs: List[Document]) -> None:

    # def stream_updater_callback(
    #     node_name: str,
    #     execution_result: Dict[str, Any]
    # ) -> None:
        # print(f"[UPDATE from {node_name}]: {execution_result}")
    
    def stream_updater_callback(step: Dict[str, Any]):
        print(f"[UPDATE]: {step}")
    
    summarizer = SummarizerAgent(llm=llm)
    steps = summarizer.summarize_stream(
        documents=docs,
        stream_updater_callback=stream_updater_callback
    )

    print("\n\n\n")
    print(steps)


if __name__ == '__main__':
    llm = init_llm('google')
    docs = load_document()
    # test_summarizer_agent(llm, docs)
    test_summarizer_agent_stream(llm, docs)
