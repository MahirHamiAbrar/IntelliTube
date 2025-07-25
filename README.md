# IntelliTube
IntelliTube is an intelligent chat system that leverages Retrieval-Augmented Generation (RAG) to help users analyze and discuss content from YouTube videos, documents, and web pages. Built with LangGraph and LangChain, it provides a conversational interface for exploring multimedia content through natural language - made for students, researchers and educators.

### Here is a working video demo of IntelliTube:
<summary>
Click to expand!
<details>

[![Vide Demo](images/intellitube_v1/homepage.png)](https://youtu.be/Z484SG5ymYc?si=KO86iOwNULbY8TW4)

</details>

</summary>

## Features
 - **YouTube Videos**: Automatically extracts and processes video transcripts
 - **Web Pages**: Loads and analyzes website content
 - **Documents**: Supports PDF, TXT, and Python files
 - **Intelligent Routing**: Automatically determines content type and processing method
 - **Persistent Chat History**: Save and resume conversations across sessions

## Agents
IntelliTube is a multi-agent system. The full system is based on multiple agents, such as -
 - **Summarizer Agent**: Summarizes given document(s) using map-reduce summarization technique.
 - **Router Agent**: Detects URLs and Local document paths and routes the user query either to a document loader or directly to a retriever to retrieve documents from vector database.
 - **Chat Agent**: Processes the user query & retrieved documents and generates an answer to respond to the user query.

The workflow of IntelliTube AI looks something like this:

<p align="center"><img src="images/main_agent_graph.png" height="auto" width="200" style="border-radius:10%"></p>


### [Summarizer Agent](agents/summrizer_agent.py)
Summarizer agent provides a qualityful summary of a given document. The graph of this agent looks something like below:

<p align="center"><img src="images/summarizer_agent_graph.png" height="auto" width="200" style="border-radius:10%"></p>

## Project Structure

```bash
intellitube
├── agents
│   ├── base_agent.py   # base agent class (all agents are built on top it)
│   ├── chat_agent      # work in progress
│   │   ├── agent.py
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── states.py
│   ├── __init__.py
│   ├── main_agent     # the agent responsible for all actions
│   │   ├── agent.py
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── states.py
│   └── summarizer_agent    # summarizes any text document
│       ├── agent.py
│       ├── __init__.py
│       ├── prompts.py
│       └── states.py
├── chatbot.py
├── data
│   ├── ...
├── images
│   ├── ...
├── __init__.py
├── llm.py  # llm initializer
├── README.md
├── requirements.txt
├── tests
│   ├── ...
├── tools
│   ├── document_loader_tools.py    # YouTube transcript/text-document/webpage loader tools
│   └── __init__.py
├── ui.py   # the streamlit frontend UI
├── utils
│   ├── cacher.py   # caches webpages & yt-transcripts on local disk for faster loading
│   ├── chat_manager.py # saves chat history and metadata
│   ├── __init__.py
│   ├── mermaid2png.py
│   ├── path_manager.py
│   ├── video_transcript.py
│   └── youtube.py
└── vector_store.py # manages the qdrant vector database
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/MahirHamiAbrar/intellitube.git
cd intellitube

# Install dependencies
pip install -r requirements.txt
```

### Setup Environment Variables

Create a `.env` file in the root directory (under `intellitube/`) and populate it with the following content:

```bash
GOOGLE_API_KEY=your-api-key
GROQ_API_KEY=your-api-key
NVIDIA_API_KEY=your-api-key
```

### Running the App

Finally when you're ready, just type this command while you're in the root directory (under `intellitube/` folder) and hit enter!

```bash
streamlit run ui.py
```

You should see a page opening in your default browser that looks like this:

![intellitube_v1_homepage.png](images/intellitube_v1/homepage.png)

If it doesn't show up for some reason, just open a browser and copy-paste this URL: (default localhost url for streamlit apps)

```bash
http://localhost:8501
```

### What's Next?

Current version of IntelliTube isn't very good at tricky questions and might generate irrelevant response for certain queries. That's why I've been working on a newer version of this agent which overcomes these limitations by following more advance architecture and retrieval techniques. Currently this agent graph architecture looks something like this:

![Intellitube AI Graph](images/intellitube_ai_graph.png)
