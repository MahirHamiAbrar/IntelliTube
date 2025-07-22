import streamlit as st
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from intellitube.utils import ChatManager
from intellitube.agents.main_agent import IntelliTubeAI


class StreamlitUI:
    page_config: dict[str, str] = {
        "page_title": "IntelliTube Chat",
        "page_icon": "🤖",
        "layout": "wide",
    }

    @property
    def chat_manager(self) -> ChatManager:
        return st.session_state.chat_manager
    
    @property
    def agent(self) -> CompiledStateGraph:
        return st.session_state.agent
    
    @property
    def chat_id(self) -> str:
        return st.session_state.chat_id
    
    @property
    def chat_messages(self) -> list[BaseMessage]:
        return st.session_state.chat_messages
    
    @chat_messages.setter
    def chat_messages(self, messages: list[BaseMessage]) -> None:
        st.session_state.chat_messages = messages
        self.chat_manager.chat_messages = messages

    def __init__(self,
        chat_manager: ChatManager,
        ai_agent: IntelliTubeAI,
    ) -> None:
        # Store in session state
        st.session_state.chat_manager = chat_manager
        st.session_state.agent = ai_agent
        st.session_state.chat_id = chat_manager.chat_id

    def setup_page(self) -> None:
        st.set_page_config(**self.page_config)
        st.title("🤖 IntelliTube Chat Assistant")
    
    def init_page(self) -> None:
        # create a placeholder for loading content
        with st.empty().container() as loading_container:
            st.info("🔄 Initializing IntelliTube backend...")

            progress_bar = st.progress(0)   # progress bar
            status_text = st.empty()
            status_text.text("Loading dependencies...")
            progress_bar.progress(20)       # update progress

            try:
                # import the backend components (this is where the loading happens)
                status_text.text("Importing IntelliTube components...")
                progress_bar.progress(50)
                
                progress_bar.progress(80)
                status_text.text("Finalizing initialization...")
                
                # store in session state
                st.session_state.backend_loaded = True
                
                progress_bar.progress(100)
                status_text.text("✅ Backend loaded successfully!")
                
                # clear loading screen and rerun
                loading_container.empty()
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Failed to load IntelliTube backend: {str(e)}")
                st.stop()
    
    def chat_page(self) -> None:
        # app header
        st.title("🤖 IntelliTube Chat Assistant")
        st.caption(f"Chat ID: {self.chat_id}")

        self.chat_page_sidebar()
        self.process_chat_page_user_input()
        self.chat_page_footer()
    
    def chat_page_sidebar(self) -> None:
        # sidebar with info
        with st.sidebar:
            st.header("💬 Chat History")
            
            # get and display chat list
            try:
                chat_list = self.chat_manager.chatlist
                if chat_list:
                    st.caption(f"Found {len(chat_list)} previous chats:")
                    
                    # sort chats by last accessed timestamp (most recent first)
                    sorted_chats = sorted(
                        chat_list.items(), 
                        key=lambda x: x[1]['last_accessed_timestamp'], 
                        reverse=True
                    )
                    
                    for chat_id, chat_info in sorted_chats:
                        # format timestamps
                        import datetime
                        created_time = datetime.datetime.fromtimestamp(chat_info['created_timestamp'])
                        last_accessed = datetime.datetime.fromtimestamp(chat_info['last_accessed_timestamp'])
                        
                        # show current chat differently
                        if chat_id == self.chat_id:
                            st.success(f"🔴 **Current Chat**")
                        else:
                            st.info(f"💭 **Chat Session**")
                        
                        # display all metadata
                        st.caption(f"**Chat ID:** `{self.chat_id}`")
                        st.caption(f"**Created:** {created_time.strftime('%Y-%m-%d at %I:%M %p')}")
                        st.caption(f"**Last Used:** {last_accessed.strftime('%Y-%m-%d at %I:%M %p')}")
                        st.divider()
                else:
                    st.caption("No previous chats found")
                    
            except Exception as e:
                st.error(f"Error loading chat history: {str(e)}")
            
            st.header("ℹ️ About")
            st.write("This is your IntelliTube chat assistant. You can:")
            st.write("- Ask questions about documents")
            st.write("- Load YouTube videos for analysis")
            st.write("- Load websites for content extraction")
            st.write("- Chat with your knowledge base")
            
            st.divider()

            if st.button("🗑️ Clear Chat"):
                self.chat_messages = []
                st.rerun()
            
            if st.button("💾 Save Chat"):
                self.chat_manager.save_chat()
                st.success("Chat saved!")
    
    def process_chat_page_user_input(self) -> None:
        # chat input
        user_input = st.chat_input("Type your message here...")

        # display existing chat messages (excluding the current processing)
        for message in self.chat_messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.write(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.write(message.content)
        
        # Process new user input
        if user_input:
            # Display user message immediately
            with st.chat_message("user"):
                st.write(user_input)
            
            # Process with agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Create HumanMessage and add to chat manager
                    usr_msg = HumanMessage(user_input)
                    self.chat_manager.add_message(usr_msg)
                    
                    # Get agent response
                    result = self.agent.invoke({"messages": self.chat_manager.chat_messages})
                    self.chat_manager.chat_messages = result["messages"]
                    
                    # Get the AI response
                    ai_msg = self.chat_manager.chat_messages[-1]
                    
                    # Add messages to session state first
                    self.chat_messages.extend([HumanMessage(user_input), ai_msg])
                    
                # Display AI response (outside spinner to avoid fading)
                st.write(ai_msg.content)
    
    def chat_page_footer(self) -> None:
        st.divider()
        st.caption("Built with IntelliTube 🚀")
