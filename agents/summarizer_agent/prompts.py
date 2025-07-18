from langchain_core.prompts import HumanMessagePromptTemplate


_map_template = """
You are a document analyst tasked with extracting the core content of a segment from a larger document.

Given the following chunk of text, identify and summarize:

1. The main **ideas** or arguments presented.
2. The key **themes** or recurring concepts.
3. Notable **facts**, **events**, or **claims** made.
4. Any distinct **perspectives**, **opinions**, or **controversies**.
5. Important **keywords** or **terms** central to understanding this chunk.

Focus on clarity and accuracy — do not interpret beyond the given chunk. Write in structured bullet points or short paragraphs, avoiding fluff, make it concise.

Here is the text chunk:
---
{context}
---
"""

_reduce_template = """
You are now synthesizing a cohesive **global summary** of the original document from a list of chunk-level bullet summaries.

Your **goal** is to pack **as much key information** as possible into **no more than 2,000 tokens** (approx. 1,400 words) so it can be inserted whole as a system prompt.

Extract and organize:

1. **Document purpose:** One sentence.
2. **Central themes:** Up to 5 bulleted theme headings (max 8 words each).
3. **Major ideas/claims:** Up to 8 numbered, concise statements (max 25 words each).
4. **Key topics:** Group related topics under 2–3 thematic headings, each with 2–4 comma-separated items.
5. **Perspectives/sides:** Briefly note up to 4 distinct viewpoints (max 20 words each).
6. **Narrative flow:** One or two sentences summarizing structure or progression.
7. **Critical keywords:** List 10–15 terms, comma-separated.

**Formatting constraints**:
- Use headings (e.g., **Purpose**, **Themes**, etc.).
- Use bullet lists or numbered lists only.
- No paragraphs longer than 2 sentences.
- Keep overall token count ≤ 2,000.

Here are the segment-level bullet summaries:
---
{docs}
---
"""

map_prompt = HumanMessagePromptTemplate.from_template(
    _map_template, input_variables=["context"]
)
"""Mapping prompt for map-reduce summarization"""

reduce_prompt = HumanMessagePromptTemplate.from_template(
    _reduce_template, input_variables=["docs"]
)
"""Reduce prompt for map-reduce summarization"""
