from intellitube.utils.chat_history import ChatHistoryManager


def new_chat_test() -> None:
    from langchain_core.messages import HumanMessage, AIMessage

    chm = ChatHistoryManager.new_chat()
    print("Chat ID:", chm._chat_id)
    # chm.add_message(HumanMessage("Hi!"))
    # chm.add_message(AIMessage("Hello there! How can I assist you?"))
    # chm.close_chat()
    print(chm.chatlist)


def load_existing_chat_test() -> None:
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    chat_id = "62dc2f34-4956-4b13-91e8-fe192b97e067"
    
    chm = ChatHistoryManager.from_chat_history(chat_id=chat_id)
    print("Chat ID:", chm.chat_id)
    chm.add_message(ToolMessage(content="This is a tool message", tool_call_id=chm.chat_id))
    chm.close_chat()


if __name__ == '__main__':
    run_test = lambda test_no: [
        new_chat_test,
        load_existing_chat_test
    ][test_no - 1].__call__()

    run_test(test_no=2)
