from langchain.chains.summarize import load_summarize_chain
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.documents import Document
from typing import List, Literal, Union
from loguru import logger


def summarize(
    llm: BaseLanguageModel, content: Union[str, Document, List[Document]],
    chain_type: Literal["stuff", "map_reduce", "refine"] = "map_reduce"
) -> str:
    if type(content) == str:
        content = [Document(page_content=content)]
    elif type(content) == Document:
        content = [content]
    
    # map_reduce_prompt = BasePromptTemplate(
    #     """Summarize the """
    # )
    
    logger.debug("generating summary...")
    chain = load_summarize_chain(llm, chain_type=chain_type)
    summary = chain.invoke(content)
    logger.debug("summary generated!")

    return summary["output_text"]
