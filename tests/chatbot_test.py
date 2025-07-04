from intellitube.chatbot import chat


def router() -> None:
    from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain.chat_models import init_chat_model
    from pydantic import BaseModel, Field
    from typing import Literal
    from dotenv import load_dotenv
    load_dotenv()

    class ModelBinaryResponse(BaseModel):
        """Your response should STRICTLY follow this format. Divide your answer into two parts: reasoning & verdict."""
        reasoning: str = Field(
            description="Divide your problem into smaller sub-problems and solve those individual problems and provide the reasoning behind your solution here."
        )
        verdict: Literal["yes", "no"] = Field(
            description="The final verdict after your reasoning. Should be one of: yes/no."
        )

    # llm = init_chat_model(model="llama-3.3-70b-versatile", model_provider="groq", temperature=0.0)
    # llm = init_chat_model(model="llama3.2:3b", model_provider="ollama", temperature=0.0)
    llm = init_chat_model(model="gemini-2.0-flash", model_provider="google_genai")

    human_message_template = (
        "Here is the user query: {query}\n"
        "Here is a short overview of a document as your context: {context}\n"
        "Now analyze the document overview and the user-query and find out"
        "if the query can be answered from the document.\n"
        "Answer from either of the two words: YES/NO."
        "DO NOT generate any additional text."
    )

    query = "Define LangGraph RAG?"
    # query = "Why sky is blue?"
    # query = "How to use Entailment ratios?"
    query = "How does Sampling-Based Detection work?"
    
    context = """This discussion centers on addressing "extrinsic hallucinations" in large language models (LLMs), where they produce incorrect or misleading information, particularly for complex topics needing external knowledge. Various evaluation benchmarks like TruthfulQA and FEVER are used to assess factual accuracy. Proposed solutions include:

- Retrieval-Augmented Generation (RAG) methods such as Self-RAG and Recite-LM that fetch relevant data before generating responses.
- Inference-time intervention techniques like ITI for truthful answers.
- Fine-tuning models on factual datasets, exemplified by FLAME and other fine-tuning approaches.
- Layer-wise verification methods such as DoLa to enhance factual consistency.
- Hallucination detection tools including SelfCheckGPT and zero-resource black-box detection.
- Selective prediction strategies allowing models to decline answering if uncertain, thus reducing misinformation spread.
- Human feedback integration in learning processes, as seen with WebGPT, for improved reliability and factual accuracy.

The post underscores continuous research efforts aimed at minimizing hallucinations in LLMs for more dependable and truthful AI systems.
    """

    chat_messages = ChatPromptTemplate.from_template(human_message_template)

    chain = chat_messages | llm.with_structured_output(ModelBinaryResponse)

    ai_msg: AIMessage = chain.invoke(
        {"query": query, "context": context}
    )
    print(ai_msg)


if __name__ == '__main__':
    chat()
    # router()
