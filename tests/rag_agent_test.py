from pydantic import BaseModel, Field
from typing_extensions import List, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

from intellitube.llm import init_llm


class QueryExpansionState(BaseModel):
    expanded_query: str = Field(description=(
        "Expand the user query for document retrieval"
    ))
    additional_queries: List[str] = Field(description=(
        "Generate 5 additional similar queries for document retrieval"
        "that can help retrieve more useful infromation"
    ))


llm = init_llm('google')    # gemini-2.5-flash


def generate_queries(
    user_message: HumanMessage, llm_custom: BaseChatModel = None
) -> QueryExpansionState:
    expander_llm = (llm_custom or llm).with_structured_output(QueryExpansionState)
    return expander_llm.invoke([user_message])


if __name__ == '__main__':
    user_message = (
        "How to stream message chunks in langgraph?"
    )

    queries = generate_queries(user_message)
    print(queries)
