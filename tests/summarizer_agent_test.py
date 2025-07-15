import asyncio
from intellitube.llm import init_llm
from intellitube.agents.summrizer_agent import SummarizerAgent
from intellitube.utils import (
    YTContentData, download_youtube_audio_or_transcript, webvtt_2_str
)

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter


async def test_summarizer_agent() -> None:
    # load a youtube video
    video_url = "https://www.youtube.com/watch?v=UF8uR6Z6KLc"
    video_data: YTContentData = download_youtube_audio_or_transcript(video_url)
    vtt_str = webvtt_2_str(vtt_file_path=video_data.transcript_path)

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    text_splits = text_splitter.split_text(vtt_str)
    split_docs = [
        Document(
            page_content=text, 
            metadata={"source": video_url, "chunk_index": n + 1}
        )
        for n, text in enumerate(text_splits)
    ]
    docs = [
        doc.page_content
        for doc in split_docs
    ]

    llm = init_llm('groq')
    summarizer = SummarizerAgent(llm=llm)

    last_step = None
    
    async for step in summarizer.agent.astream(
        input={"documents": docs},
        config={"recursion_limit": 30},
    ):
        print(list(step.keys()))
        last_step = step
    
    print(last_step)


if __name__ == '__main__':
    asyncio.run(test_summarizer_agent())
