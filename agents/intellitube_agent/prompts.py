from langchain_core.prompts import SystemMessagePromptTemplate

_multi_query_prompt_template = """
You are an intelligent query generation agent designed to enhance retrieval quality in a RAG-based system.

Given a user question and a brief summary of the source document, your job is to:
1. Rewrite the user query to be more precise, clear, and unambiguous.
2. Generate multiple diverse sub-queries that capture various aspects or phrasings of the original query. These sub-queries should target different possible ways the information might appear in the document.

Summary of the document:
{summary}

Your output should include:
- "rewritten_query": A refined and clarified version of the original user query.
- "multi_query": A list of diverse sub-queries that improve semantic coverage during retrieval.
"""

multi_query_prompt = SystemMessagePromptTemplate.from_template(
    _multi_query_prompt_template, input_variables=["summary"]
)
"""Prompt for generating multi-querys & rewritten query"""
