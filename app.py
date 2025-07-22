from intellitube.llm import init_llm
from intellitube.utils import ChatManager
from intellitube.vector_store import VectorStoreManager

from intellitube.ui.streamlit_ui import StreamlitUI
from intellitube.agents.main_agent import IntelliTubeAI


def run_app() -> None:
    # initialize an llm
    llm = init_llm(model_provider='google')
    
    # initialize the chat manager
    chatman = ChatManager.new_chat()
    
    # initialize vector store
    vsman = VectorStoreManager()

    # initialize the agent
    ai_agent = IntelliTubeAI(
        llm=llm, chat_manager=chatman,
        vector_store_manager=vsman,
    )

    # initialize the UI
    ui = StreamlitUI(
        chat_manager=chatman, ai_agent=ai_agent
    )
    ui.launch()


if __name__ == '__main__':
    run_app()
