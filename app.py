import os
from typing import Union
from intellitube.llm import init_llm
from intellitube.utils import ChatManager
from intellitube.vector_store import VectorStoreManager

from intellitube.ui.streamlit_ui import StreamlitUI
from intellitube.agents.main_agent import IntelliTubeAI


def init_function() -> Union[ChatManager, IntelliTubeAI]:
    # initialize an llm
    llm = init_llm(model_provider='google')
    
    # initialize the chat manager
    chatman = ChatManager.new_chat()
    
    # initialize vector store
    vsman = VectorStoreManager(
        path_on_disk=chatman.chat_dirpath,
        collection_path_on_disk=os.path.join(chatman.chat_dirpath, "collection"),
        collection_name=chatman.chat_id,
    )

    # initialize the agent
    ai_agent = IntelliTubeAI(
        llm=llm, chat_manager=chatman,
        vector_store_manager=vsman,
    )

    # ai_agent.cli_chat_loop()
    return chatman, ai_agent

def run_app() -> None:
    # initialize the UI
    # ui = StreamlitUI(init_function=init_function)
    ui = StreamlitUI()
    ui.launch()


if __name__ == '__main__':
    run_app()
