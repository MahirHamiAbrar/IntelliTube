import os
import uuid
import shutil
import tempfile
from pathlib import Path

from loguru import logger
from typing import Any, Dict, List, Literal, Union, Optional

from yt_dlp import YoutubeDL
from pydantic import BaseModel

from .cacher import Cacher


class YTContentData(BaseModel):
    type: Literal['text', 'audio', 'both']
    transcript_path: Optional[str] = None
    audio_path: Optional[str] = None

    def model_post_init(self, context: Any) -> None:
        if not self.transcript_path and not self.audio_path:
            raise AttributeError("Both `transcript_path` and `audio_path` cannot be empty!")
        
        elif self.type == 'audio':
            if not self.audio_path:
                raise AttributeError("`type` was set to `audio` but `audio_path` is empty!")
            
            if not os.path.exists(self.audio_path):
                raise FileNotFoundError('Provided `audio_path` does not exist!')
        
        elif self.type == 'text':
            if not self.transcript_path:
                raise AttributeError("`type` was set to `text` but `transcript_path` is empty!")
            
            if not os.path.exists(self.transcript_path):
                raise FileNotFoundError('Provided `transcript_path` does not exist!')
        
        if self.transcript_path and self.audio_path:
            self.type = 'both'
        
        return super().model_post_init(context)


def search_youtube(query: str, max_results=5) -> List[Dict[str, Any]]:
    """
    Searches YouTube using yt-dlp and returns a list of video metadata.

    Args:
        query (str): The search query.
        max_results (int): Number of top results to return.

    Returns:
        list: A list of dictionaries containing video metadata (title, id, url, duration, etc.)
    """
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
        'extract_flat': True,
        'force_generic_extractor': False,
    }

    search_url = f"ytsearch{max_results}:{query}"

    with YoutubeDL(ydl_opts) as ydl:
        info: Dict[str, Any] = ydl.extract_info(search_url, download=False)
        return [
            {
                'title': entry.get('title'),
                'id': entry.get('id'),
                'url': f"https://www.youtube.com/watch?v={entry.get('id')}",
                'duration': entry.get('duration'),
                'channel': entry.get('uploader'),
            }
            for entry in info.get('entries', [])
        ]


def download_youtube_content(
    url: str,
    output_dir: str,
    suffix: Literal['mp3', 'vtt'],
    config: Dict[str, Any]
) -> Union[str, None]:
    
    # create a output path
    unq_fname = str(uuid.uuid4())
    output_path = os.path.join(output_dir, unq_fname) + '.' + suffix

    with tempfile.TemporaryDirectory(dir=output_dir) as tempdir:
        temp_file = os.path.join(tempdir, f"{unq_fname}.%(ext)s")
        config['outtmpl'] = temp_file

        with YoutubeDL(config) as ydl:
            try:
                # download the transcript
                ydl.download([url])

                downloaded_fp = None

                for filename in os.listdir(tempdir):
                    if filename.startswith(unq_fname):
                        downloaded_fp = os.path.join(tempdir, filename)
                        break
                
                if not downloaded_fp or not os.path.exists(downloaded_fp):
                    raise Exception('Download was unsuccessful!')
                
                shutil.move(downloaded_fp, output_path)

            except Exception as e:
                logger.error(f"Error encountered: {e}")
                return None
    
    return output_path


def download_youtube_audio_or_transcript(
    video_url: str,
    preferred_output: Literal['audio', 'transcript', 'both'] = 'transcript',
    cache_dir: str = 'cache/youtube/cache',
    output_dir: str = 'cache/youtube/downloads',
    use_cache: bool = True,
) -> YTContentData:
    
    cache = Cacher(video_url, 'json', cache_dir)
    cache_data: YTContentData = None
    
    # TODO: Delete old cache
    # TODO: SOLVE ISSUE: when use_cache=False, it recreates the cache from scratch

    if cache.cache_file_exists() and use_cache:
        logger.debug('Cache exists, validating cache...')

        cache_data = YTContentData(**cache.load_cache())
        cache_valid = True

        if preferred_output in ['transcript', 'both'] and not cache_data.transcript_path:
            cache_valid = False
        
        if preferred_output in ['audio', 'both'] and not cache_data.audio_path:
            cache_valid = False

        if cache_valid:
            logger.debug('Cache contains the requested data. Using cache.')
            return cache_data
        logger.debug('Current cache DOES NOT contain newly requested data. Downloading it now...')

    # first create output dirs, if they already don't exist
    os.makedirs(output_dir, exist_ok=True)

    audio_path = None
    transcript_path = None

    if preferred_output in ['transcript', 'both']:
        logger.debug('Downloading transcript...')
        transcript_path = download_youtube_content(
            url=video_url,
            output_dir=output_dir,
            suffix='vtt',
            config={
                'writesubtitles': True,
                'writeautomaticsub': True,
                'skip_download': True,
                'quiet': True,
            }    
        )
    
    if preferred_output in ['audio', 'both']:
        logger.debug('Downloading audio...')
        audio_path = download_youtube_content(
            url=video_url,
            output_dir=output_dir,
            suffix='mp3',
            config={
                'format': 'bestaudio/best',
                'quiet': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'postprocessor_args': [
                    # Set sample rate to 44.1kHz, which is CD quality and recommended
                    '-ar', '44100'
                ],
                'prefer_ffmpeg': True,
            }
        )
    mhA_G1T_p@$$!
    if cache_data:
        transcript_path = transcript_path or cache_data.transcript_path
        audio_path = audio_path or cache_data.audio_path
    
    data = YTContentData(
        type = 'text' if preferred_output == 'transcript' else preferred_output,
        transcript_path = transcript_path,
        audio_path = audio_path,
    )
    cache.save_cache(data.model_dump())
    return data
