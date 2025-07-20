import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Page config
st.set_page_config(
    page_title="IntelliTube Chat",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for backend loading
if "backend_loaded" not in st.session_state:
    st.session_state.backend_loaded = False
    st.session_state.chat_messages = []

# Show loading screen while backend initializes
if not st.session_state.backend_loaded:
    st.title("🤖 IntelliTube Chat Assistant")
    
    # Create a placeholder for loading content
    loading_container = st.empty()
    
    with loading_container.container():
        st.info("🔄 Initializing IntelliTube backend...")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Update progress
        status_text.text("Loading dependencies...")
        progress_bar.progress(20)
        
        try:
            # Import the backend components (this is where the loading happens)
            status_text.text("Importing IntelliTube components...")
            progress_bar.progress(50)
            
            from intellitube.data.old_codes.intellitube_ai_draft import (
                chat_manager, agent
            )
            
            progress_bar.progress(80)
            status_text.text("Finalizing initialization...")
            
            # Store in session state
            st.session_state.chat_manager = chat_manager
            st.session_state.agent = agent
            st.session_state.chat_id = chat_manager.chat_id
            st.session_state.backend_loaded = True
            
            progress_bar.progress(100)
            status_text.text("✅ Backend loaded successfully!")
            
            # Small delay to show completion
            import time
            time.sleep(0.5)
            
            # Clear loading screen and rerun
            loading_container.empty()
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Failed to load IntelliTube backend: {str(e)}")
            st.stop()

# Main app (only shown after backend is loaded)
else:
    # Get components from session state
    chat_manager = st.session_state.chat_manager
    agent = st.session_state.agent
    
    # App header
    st.title("🤖 IntelliTube Chat Assistant")
    st.caption(f"Chat ID: {st.session_state.chat_id}")

    # Sidebar with info
    with st.sidebar:
        st.header("💬 Chat History")
        
        # Get and display chat list
        try:
            chat_list = chat_manager.chatlist
            if chat_list:
                st.caption(f"Found {len(chat_list)} previous chats:")
                
                # Sort chats by last accessed timestamp (most recent first)
                sorted_chats = sorted(
                    chat_list.items(), 
                    key=lambda x: x[1]['last_accessed_timestamp'], 
                    reverse=True
                )
                
                for chat_id, chat_info in sorted_chats:
                    # Format timestamps
                    import datetime
                    created_time = datetime.datetime.fromtimestamp(chat_info['created_timestamp'])
                    last_accessed = datetime.datetime.fromtimestamp(chat_info['last_accessed_timestamp'])
                    
                    # Show current chat differently
                    if chat_id == st.session_state.chat_id:
                        st.success(f"🔴 **Current Chat**")
                    else:
                        st.info(f"💭 **Chat Session**")
                    
                    # Display all metadata
                    st.caption(f"**ID:** {chat_id[:8]}...")
                    st.caption(f"**Full ID:** `{chat_id}`")
                    st.caption(f"**Created:** {created_time.strftime('%Y-%m-%d at %I:%M %p')}")
                    st.caption(f"**Last Used:** {last_accessed.strftime('%Y-%m-%d at %I:%M %p')}")
                    # st.caption(f"**Created Timestamp:** {chat_info['created_timestamp']}")
                    # st.caption(f"**Last Accessed Timestamp:** {chat_info['last_accessed_timestamp']}")
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
            st.session_state.chat_messages = []
            chat_manager.chat_messages = []
            st.rerun()
        
        if st.button("💾 Save Chat"):
            chat_manager.save_chat()
            st.success("Chat saved!")

    # Chat input
    user_input = st.chat_input("Type your message here...")

    # Display existing chat messages (excluding the current processing)
    for message in st.session_state.chat_messages:
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
                chat_manager.add_message(usr_msg)
                
                # Get agent response
                result = agent.invoke({"messages": chat_manager.chat_messages})
                chat_manager.chat_messages = result["messages"]
                
                # Get the AI response
                ai_msg = chat_manager.chat_messages[-1]
                
                # Add messages to session state first
                st.session_state.chat_messages.extend([HumanMessage(user_input), ai_msg])
                
            # Display AI response (outside spinner to avoid fading)
            st.write(ai_msg.content)

    # Footer
    st.divider()
    st.caption("Built with IntelliTube 🚀")