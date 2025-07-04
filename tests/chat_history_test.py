from intellitube.utils.chat_history import ChatHistoryManager


if __name__ == '__main__':
    from langchain_core.messages import HumanMessage, AIMessage

    chm = ChatHistoryManager.new_chat()
    print("Chat ID:", chm._chat_id)
    chm.add_message(HumanMessage("Hi!"))
    chm.add_message(AIMessage("Hello there! How can I assist you?"))
    chm.close_chat()
    # chat_list = chm.list_chats()
    # print(chat_list)

    # 62dc2f34-4956-4b13-91e8-fe192b97e067

