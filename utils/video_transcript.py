import re
from typing import Any, Dict, List, Optional, Union
from langchain_core.documents import Document


def webvtt_2_json(
    vtt_content: Optional[str] = None,
    vtt_file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Converts a WEBVTT file/content to a JSON object.

    Args:
        vtt_content (Optional[str], optional): WEBVTT File content as a string. Defaults to None.
        vtt_file_path (Optional[str], optional): WEBVTT file path. Defaults to None.

    Raises:
        ValueError: If both of the arguments found empty.
    
    **NOTE: If both values are provided, `vtt_content` is used and `vtt_file_path` is ignored.**

    Returns:
        Dict[str, Any]: A dictionary containing metadata & captions.
    
    # Example Return Output:
    ```python
    {
        "metadata": {
            'type': 'WEBVTT', 
            'Kind': 'captions', 
            'Language': 'en'
        },
        "captions": list(
            {
                "start": "00:00:00.000",
                "end": "00:00:03.360",
                "text": "caption text"
            },
            ...
        )
    }
    ```
    """
    
    if not vtt_file_path and not vtt_content:
        raise ValueError("One of `vtt_file_path` and `vtt_content` must be provided. Found both blank!")
    
    lines: List[str] = (vtt_content or "").split("\n")

    if not lines[0]:
        with open(vtt_file_path, 'r') as file:
            lines = file.read().strip().split("\n")
    
    metadata = {}
    json_data = []

    for line in lines:
        try:
            if not line: continue

            M = re.match("\d+:\d+:.+ --> \d+:\d+:.+", line)
            # print(line, M)

            if M:
                time_range = line.split("-->")
                json_data.append({
                    "start": time_range[0].strip(),
                    "end": time_range[1].strip(),
                })
                # print(f"Match: {M.start(), M.end(), M.string}")
            else:
                if not json_data and not M:
                    if line != "WEBVTT":
                        data = line.split(':')
                        metadata[data[0].strip()] = data[1].strip()
                    else:
                        metadata["type"] = "WEBVTT"
                else:
                    json_data[-1]["text"] = line
        
        except Exception as e:
            print(f"ERROR: {e}")
    
    return {
        "metadata": metadata,
        "captions": json_data
    }


def webvtt_2_langchain_documents(
    vtt_content: Optional[Union[str, dict]] = None,
    vtt_file_path: Optional[str] = None,
) -> List[Document]:
    """Converts WEBVTT file/content to a `list` of LangChain `Document` Objects.

    Args:
        vtt_content (Optional[str | dict], optional): WEBVTT File content as a `str` or a `dict`. Defaults to None.
        vtt_file_path (Optional[str], optional): WEBVTT file path. Defaults to None.

    Returns:
        List[Document]: `list` of LangChain `Document` Objects.
    """
    
    vtt_json: dict
    documents: List[Document] = []

    if type(vtt_content) == dict:
        vtt_json = vtt_content
    else:
        vtt_json = webvtt_2_json(vtt_content, vtt_file_path)

    for caption in vtt_json["captions"]:
        documents.append(Document(
            page_content=caption['text'],
            metadata={
                **vtt_json["metadata"],
                "start": caption["start"],
                "end": caption["end"],
            }
        ))
    
    return documents
