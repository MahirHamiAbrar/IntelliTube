from intellitube.utils import webvtt_2_json, webvtt_2_langchain_documents
from intellitube.utils import YTContentData, download_youtube_audio_or_transcript


if __name__ == '__main__':
    url = 'https://www.youtube.com/watch?v=W3I3kAg2J7w&t=231s'
    
    data: YTContentData = download_youtube_audio_or_transcript(
        video_url=url
    )
    print("Downloaded Data: ", data, end='\n\n')

    json_data = webvtt_2_json(vtt_file_path=data.transcript_path)
    print("JSON Data:", json_data, end='\n\n')

    data = webvtt_2_langchain_documents(vtt_content=json_data)
    print(data[0])
    print(len(data))