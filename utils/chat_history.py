"""Create and maintain chat history"""
import os
import json
import uuid
from datetime import datetime
from loguru import logger
from typing import Any, List, Dict, Self, TypedDict, Union, Literal, Optional

from pydantic import BaseModel
from langchain_core.messages import BaseMessage


class ChatInfo(TypedDict):
    chat_id: str
    created_timestamp: float
    last_accessed_timestamp: float


class Chat(BaseModel):
    messages: List[BaseMessage]


class ChatHistoryManager:
    _save_dir: str = "test_data/chat_history"
    _chats_dir: str = "chats"   # make it a folder, container of: chat-messages + vector-database
    _chat_list_fname: str = "chat_list.json"
    _chat_list_fp: str = None
    _chat_id: str = ""

    _chat: Chat = Chat(messages=[])
    _chat_list: Dict[str, ChatInfo] = {}
    
    @property
    def save_dir(self) -> str:
        return self._save_dir
    
    @save_dir.setter
    def save_dir(self, path: str) -> None:
        os.makedirs(os.path.join(path, self._chats_dir), exist_ok=True)
        self._save_dir = path
        self._chat_list_fp = os.path.join(self._save_dir, self._chat_list_fname)
        
        if not os.path.exists(self._chat_list_fp):
            with open(self._chat_list_fp, 'w') as chat_list_file:
                json.dump(self._chat_list, chat_list_file)
    
    @property
    def chatlist(self) -> Dict[str, ChatInfo]:
        if not self._chat_list:
            self._chat_list = self.load_chatlist()
        return self._chat_list
    
    @staticmethod
    def new_chat(chat_id: Optional[str] = None, save_dir: Optional[str] = None) -> Self:
        chat_id = chat_id or str(uuid.uuid4())
        _manager = ChatHistoryManager(save_dir, chat_id=chat_id)

        if _manager.chatlist.get(chat_id, None):
            raise ValueError(f"Given chat_id={chat_id} already exists. Please provide an unique chat id.")
        
        _dt_now_ts = datetime.timestamp(datetime.now())
        _manager.chatlist[chat_id] = ChatInfo(
            chat_id=chat_id,
            created_timestamp=_dt_now_ts,
            last_accessed_timestamp=_dt_now_ts,
        )

        return _manager
    
    # @staticmethod
    def from_chat_history(self, chat_id: str) -> Self:
        with open(self.get_chat_location(chat_id), 'r') as chat_file:
            self._chat = Chat(**json.load(chat_file))
        return self
    
    def __init__(self,
        save_dir: Optional[str] = None,
        chat_id: Optional[str] = None,
        chatlist: Optional[Dict[str, ChatInfo]] = None,
        chat: Optional[Chat] = None
    ) -> None:
        self.save_dir = save_dir or self._save_dir

        if chat_id: self._chat_id = chat_id
        if chatlist: self._chat_list = chatlist
        if chat: self._chat = chat
    
    def vdb_exists(self) -> bool:
        pass

    def load_chatlist(self) -> Dict[str, ChatInfo]:
        with open(self._chat_list_fp, 'r') as chat_list_file:
            return json.load(chat_list_file)
    
    def refresh_chatlist(self) -> Dict[str, ChatInfo]:
        self._chat_list = self.load_chatlist()
        return self._chat_list
    
    def save_chatlist(self) -> None:
        with open(self._chat_list_fp, 'w') as chat_list_file:
            json.dump(self._chat_list, chat_list_file, indent=4)
    
    def get_chat_location(self, chat_id: str) -> str:
        return os.path.join(self.save_dir, self._chats_dir, f"{chat_id}.json")

    def save_chat(self, indent: int = 4) -> None:
        with open(self.get_chat_location(self._chat_id), 'w') as chat_file:
            json.dump(self._chat.model_dump(), chat_file, indent=indent)
    
    def close_chat(self) -> None:
        self.save_chat()
        self.save_chatlist()
    
    def chat(self) -> Chat:
        return self._chat
    
    def add_message(self, message: BaseMessage) -> None:
        self._chat.messages.append(message)
