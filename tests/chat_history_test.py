from utils.chat_manager import ChatManager


def new_chat_test() -> None:
    from langchain_core.messages import HumanMessage, AIMessage

    chm = ChatManager.new_chat()
    print("Chat ID:", chm._chat_id)
    chm.add_message(HumanMessage("Hi!"))
    chm.add_message(AIMessage("Hello there! How can I assist you?"))
    chm.end_chat()
    print(chm.chatlist)


def load_existing_chat_test() -> None:
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    chat_id = "95c6a153-f7eb-4e22-a075-0e24f18c5b44"
    
    chm = ChatManager.from_chat_history(chat_id=chat_id)
    print("Chat ID:", chm.chat_id)
    chm.add_message(ToolMessage(content="This is a tool message", tool_call_id=chm.chat_id))
    chm.end_chat()


if __name__ == '__main__':
    run_test = lambda test_no: [
        new_chat_test,
        load_existing_chat_test
    ][test_no - 2].__call__()

    run_test(test_no=1)
