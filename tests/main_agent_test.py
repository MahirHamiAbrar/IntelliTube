user_message = lambda n: [
    (
        "How to implement an agentic rag system according to this website?\n"
        "https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/"
    ),
    (
        "What's that python lib he used in this video to chunk text?\n"
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ),
    (
        "plz get the steps for multi-modal langchain agent from this\n"
        "https://www.langchain.com/blog/multimodal-agent"
    ),
    (
        "Check my saved pdf here for the instructions on vector DB evals:\n"
        "/home/mhabrar/docs/vdb-eval-guide.pdf"
    ),
    (
        "How to deploy this model in Huggingface spaces? See here:\n"
        "https://huggingface.co/spaces/yuntian-deng/ChatGPT"
    ),
    (
        "Extract the todo steps from this colab:\n"
        "https://colab.research.google.com/drive/1xxxyyyzzz"
    ),
    (
        "what r the limits of llama3-8b according to the blog here\n"
        "https://blog.groq.com/llama3"
    ),
    (
        "The video explains streaming eval but I don't get the part at 3:52\n"
        "https://youtu.be/_eSYjZ2x9rM"
    ),
    (
        "summarize whatever's in this txt file about finetuning\n"
        "./data/private_notes/finetune-steps.txt"
    ),
    (
        "can you get the eval matrix he used from this github?\n"
        "https://github.com/hwchase17/langchain/blob/master/evals/README.md"
    ),
    (
        "I forgot what I wrote but check this md file\n"
        "/mnt/data/notes/aug-agent.md"
    ),
    (
        "What are the differences btwn semantic and hybrid search from the docs?\n"
        "https://www.pinecone.io/learn/hybrid-search/"
    ),
    (
        "Umm can u check this site and tell me how they used reranking?\n"
        "https://cohere.com/blog/re-ranking"
    ),
    (
        "get the flow diagram from this repo:\n"
        "https://github.com/langchain-ai/langgraph/tree/main/examples/rag"
    ),
    (
        "See the prompt examples given in the 2nd half of this notebook:\n"
        "https://colab.research.google.com/drive/abcd1234"
    ),
    (
        "compare llamaindex vs langchain from the last section here\n"
        "https://blog.langchain.dev/langchain-vs-llamaindex/"
    ),
    (
        "how does that docker file setup work again? It's in this local folder\n"
        "~/projects/ragstack/"
    ),
    (
        "extract the retry logic from the python file in this directory:\n"
        "./scripts/error_handler.py"
    ),
    (
        "check the HuggingFace card of this model:\n"
        "https://huggingface.co/meta-llama/Meta-Llama-3-8B"
    ),
    (
        "Please find what vectorstore works best for long docs acc to this:\n"
        "https://qdrant.tech/documentation/overview/"
    ),
    (
        "he mentioned the chunk sizes somewhere here but idk where\n"
        "https://www.youtube.com/watch?v=jN3WcYUXIpc"
    ),
    (
        "check the config file at /etc/vector/config.yaml for which DB is used"
    ),
    (
        "so i wanna get the model load steps from this md:\n"
        "./readme/model-loading.md"
    ),
    (
        "get the 3rd example's output screenshot from this ipynb:\n"
        "./notebooks/fewshot_agents.ipynb"
    ),
    (
        "explain the observability module used here pls\n"
        "https://docs.langchain.com/docs/modules/model_io/observability"
    ),
    (
        "I saved a webarchive at this path. see what retriever they're using:\n"
        "~/archive/langchain-retrieval.webarchive"
    ),
    (
        "find the quote about memory bottleneck from this reddit thread:\n"
        "https://www.reddit.com/r/LocalLLaMA/comments/xxxyyy/"
    ),
    (
        "plz get the list of agents from the table shown here:\n"
        "https://python.langchain.com/v0.1/docs/modules/agents/"
    ),
    (
        "whats the key takeaway from the last para in this article?\n"
        "https://sebastianraschka.com/blog/2023/langchain-llm-agent.html"
    ),
    (
        "what's the best rag architechture acc to this blog?\n"
        "https://www.rungalileo.io/blog/agentic-rag-is-the-future"
    ),
    (
        "I think i have the prompt inside /home/user/prompts/rag.txt"
    ),
    (
        "summarize whatever this guy's saying abt eval\n"
        "https://youtu.be/BQ4U_M6d7VI"
    ),
    (
        "he used guidance lib or smthng, see this:\n"
        "https://github.com/microsoft/guidance"
    ),
    (
        "idk where it is but I saved a backup config here:\n"
        "~/backups/langchain/config.json"
    ),
    (
        "extract the doc title and first h2 from this html file:\n"
        "./webscrapes/langchain-agent.html"
    ),
    (
        "what r the key differences in the retriever logic from here?\n"
        "https://github.com/langchain-ai/langchain/blob/master/langchain/retrievers/base.py"
    ),
    (
        "Is it okay to use Postgres as vector DB acc to this post?\n"
        "https://ankane.org/pgvector"
    ),
    (
        "How does it even work lol\n"
        "https://huggingface.co/blog/llama-3"
    ),
    (
        "See this link n tell me if it uses langgraph:\n"
        "https://github.com/jerryjliu/llama_index"
    ),
    (
        "check the second half of this yt vid. He explains the eval framework\n"
        "https://youtu.be/qw4rdzRKieY"
    ),
    (
        "the code's in my repo, path: ./intellitube/agents/main.py"
    ),
    (
        "I wrote the idea somewhere in draft.txt... just quote it pls"
    ),
    (
        "fetch the steps from that PDF in Downloads\n"
        "~/Downloads/rag-architechture.pdf"
    ),
    (
        "Find how he chunked YouTube transcript in the notebook\n"
        "https://colab.research.google.com/drive/xyz987"
    ),
    (
        "Get the pros and cons of Gemini vs GPT from this video\n"
        "https://www.youtube.com/watch?v=XyzAi1z3GiM"
    ),
    (
        "i left a note inside this file i think:\n"
        "~/docs/rag_agent_notes.md"
    ),
    (
        "what tools did he add in the agent toolbox? check this:\n"
        "https://www.langchain.com/use-cases/agents"
    ),
    (
        "summarise the YAML config file located at:\n"
        "./configs/langflow.yaml"
    ),
    (
        "Compare llama3 and granite8b from this HuggingFace blog:\n"
        "https://huggingface.co/blog/granite-llama3-comparison"
    ),
    (
        "get that diagram he showed at 2:11 of this vid\n"
        "https://youtu.be/_AGDzy68Q90"
    ),
][n - 1]


import json
from pathlib import Path
from intellitube.main_agent import extract_query

def create_unique_path(folder: Path | str, filename: Path | str) -> Path:
    if not isinstance(folder, Path):
        folder = Path(folder)
    
    if not isinstance(filename, Path):
        filename = Path(filename)

    n = 1
    stem, suffix = filename.stem, filename.suffix
    path = folder / filename

    while path.exists():
        filename = stem + f"{n:05d}" + suffix
        path = folder / filename
        n += 1
    return path


if __name__ == '__main__':

    messages = []

    try:
        # for i in range(1, 51):
        for i in range(20, 30):
            message = user_message(i)
            query_extractor_response = extract_query(message)

            messages.append({
                "user": message,
                "extractor": query_extractor_response.model_dump()
            })

            print("=== " * 30)
            print(f'User Message: {message}')
            print('---' * 30)
            print(query_extractor_response, end='\n\n')
    except Exception as e:
        print(e)
    
    path = create_unique_path(
        "test_data/queryextractor", "extractor_messages_gemini.json"
    )
    with open(path, 'w') as file:
        json.dump(messages, file, indent=4)
