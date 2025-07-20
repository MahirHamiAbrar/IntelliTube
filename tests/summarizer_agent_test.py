import asyncio
from intellitube.llm import init_llm
from intellitube.agents import SummarizerAgent, SummarizerAgentState
from intellitube.utils import (
    YTContentData, download_youtube_audio_or_transcript, webvtt_2_str
)

from typing import Any, Dict, List
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_text_splitters import CharacterTextSplitter

"""

what gives life meaning? "https://www.youtube.com/watch?v=UF8uR6Z6KLc"

at what age did he go to college?

what is the economic impacts of climate change? /home/mhabrar/Apps/miniconda3/envs/pyenv/lib/python3.13/intellitube/test_data/dummy_doc.txt

how to install wine? https://wine.htmlvalidator.com/install-wine-on-arch-linux.html

"""


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
    def stream_updater_callback(step: Dict[str, Any]):
        print(f"[UPDATE]: {list(step.keys())}")
    
    summarizer = SummarizerAgent(llm=llm)
    results = summarizer.stream_summarize(
        documents=docs,
        stream_updater_callback=stream_updater_callback
    )

    print("\n\n\n")
    print(results)  # steps, SummarizerAgentState

async def atest_summarizer_agent_stream(llm: BaseChatModel, docs: List[Document]) -> None:
    def stream_updater_callback(step: Dict[str, Any]):
        print(f"[UPDATE]: {list(step.keys())}")
    
    summarizer = SummarizerAgent(llm=llm)
    results = await summarizer.stream_asummarize(
        documents=docs,
        stream_updater_callback=stream_updater_callback
    )

    print("\n\n\n")
    print(results)  # steps, SummarizerAgentState


if __name__ == '__main__':
    llm = init_llm('google')
    docs = load_document()

    run_func = lambda n: [
        test_summarizer_agent,
        test_summarizer_agent_stream,
        lambda llm, docs: asyncio.run(atest_summarizer_agent_stream(llm, docs)),
    ][n - 1].__call__(llm, docs)

    run_func(3)
