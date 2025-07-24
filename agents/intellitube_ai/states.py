from typing_extensions import Annotated, Literal, Optional, TypedDict

class QueryExtractorState(TypedDict):
    instruction: Annotated[str, ..., (
        "The user's instruction quoted word-for-word with any URLs or Paths removed.\n"
        "Preserve the casing, punctuation, and wording. Do NOT fix typos or grammar."
    )]

    analysis: Annotated[str, ..., (
        "Aanalyze the RAW data and describe what the user actually meant in one sentence."
        "Be clear and concise about the user's intention."
    )]

    url: Annotated[Optional[str], ..., (
        "The URL or local path provided by the user, if any."
        "Should be extracted separately from the user-query."
        "If there is no URL or file path, leave this as null (do not fabricate one).\n"
        "Example: 'https://example.com/page', 'C:/Documents/myfile.txt', './notes.md'"
    )] = None

    urlof: Annotated[
        Optional[Literal["youtube_video", "website", "document"]], ..., (
        "The type of content the `url` field refers to:\n"
        "- 'youtube_video': if it's a YouTube video link\n"
        "- 'website': for general websites or web pages\n"
        "- 'document': for file paths (like .txt, .pdf, .md, etc.)\n"
        "If no URL/path is provided, this should be null."
    )] = None
