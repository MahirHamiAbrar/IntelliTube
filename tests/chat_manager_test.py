from intellitube.utils.chat_manager import ChatManager


def new_chat_test() -> None:
    from langchain_core.messages import HumanMessage, AIMessage

    chatman = ChatManager.new_chat()
    print("Chat ID:", chatman._chat_id)
    chatman.add_message(HumanMessage("Hi!"))
    chatman.add_message(AIMessage("Hello there! How can I assist you?"))
    chatman.save_chat()
    print(chatman.chatlist)

    import os
    os.system(f"code {chatman.chat_filepath}")


def load_existing_chat_test() -> None:
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    chat_id = "1bd8f496-61a1-4947-9652-f5e86bb8d07f"
    
    chatman = ChatManager.from_chat_history(chat_id=chat_id)
    print("Chat ID:", chatman.chat_id)
    chatman.add_message(ToolMessage(content="This is a tool message", tool_call_id=chatman.chat_id))
    chatman.save_chat()


if __name__ == '__main__':
    run_test = lambda test_no: [
        new_chat_test,
        load_existing_chat_test
    ][test_no - 1].__call__()

    run_test(test_no=2)
