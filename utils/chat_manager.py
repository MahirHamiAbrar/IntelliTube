"""Create and maintain chat history"""
import json
import uuid
import shutil
from pathlib import Path
from loguru import logger
from datetime import datetime
from typing_extensions import Any, List, Dict, TypedDict, Optional, Union

from pydantic import BaseModel
from langchain_core.messages import BaseMessage


class ChatInfo(TypedDict):
    chat_id: str
    created_timestamp: float
    last_accessed_timestamp: float


class Chat(BaseModel):
    messages: List[BaseMessage]
    additional_data: Dict[str, Any] = {}


class ChatManager:
    _root_dir: Path = Path("test_data/chat_history")
    _chats_dir: str = "chats"   # container of: chat-messages + vector-database
    _chat_id: str = ""
    
    _chat_filename: str = "chat_messages.json"
    _chatlist_filename: str = "chatlist.json"
    _chatlist_filepath: Path = None
    
    _chat_dirpath: Path = None
    _chat_filepath: Path = None

    _chat: Chat = Chat(messages=[])
    _chat_list: Dict[str, ChatInfo] = {}
    
    @property
    def root_dir(self) -> Path:
        if not isinstance(self._root_dir, Path):
            self._root_dir = Path(self._root_dir)
        return self._root_dir
    
    @root_dir.setter
    def root_dir(self, path: Union[Path, str]) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        
        # create the folder along with the parent directories if they dont exist already
        dirpath = path / self.chats_dir
        dirpath.mkdir(parents=True, exist_ok=True)
        
        # set the paths
        self._root_dir = path
        self._chatlist_filepath = self.root_dir / self.chatlist_filename
        
        # create `chat_list.json` if it does not exist already
        if not self.chatlist_filepath.exists():
            self.chatlist_filepath.write_text(
                json.dumps(self._chat_list, indent=4)
            )
    
    @property
    def chats_dir(self) -> str:
        return self._chats_dir
    
    @property
    def chat_id(self) -> str:
        if not self._chat_id:
            self._chat_id = str(uuid.uuid4())
        return self._chat_id
    
    @property
    def chat_filename(self) -> str:
        return self._chat_filename
    
    @property
    def chatlist_filename(self) -> str:
        return self._chatlist_filename
    
    @property
    def chatlist_filepath(self) -> Path:
        if not isinstance(self._chatlist_filepath, Path):
            self._chatlist_filepath = Path(self._chatlist_filepath)
        return self._chatlist_filepath
    
    @property
    def chat_dirpath(self) -> Path:
        if not self._chat_dirpath:
            self._chat_dirpath = self.root_dir / self.chats_dir / self.chat_id
        return self._chat_dirpath
    
    @property
    def chat_filepath(self) -> Path:
        if not self._chat_filepath:
            self._chat_filepath = self.chat_dirpath / self.chat_filename
        return self._chat_filepath
    
    @property
    def chatlist(self) -> Dict[str, ChatInfo]:
        if not self._chat_list:
            self._chat_list = self.load_chatlist()
        return self._chat_list
    
    @property
    def chat(self) -> Chat:
        return self._chat
    
    @property
    def chat_messages(self) -> List[BaseMessage]:
        return self.chat.messages
    
    @chat_messages.setter
    def chat_messages(self, messages: List[BaseMessage]) -> None:
        self._chat.messages = messages
    
    @staticmethod
    def new_chat(chat_id: Optional[str] = None, root_dir: Optional[str] = None) -> 'ChatManager':
        _manager = ChatManager(root_dir, chat_id=chat_id)

        if _manager.chatlist.get(_manager.chat_id, None):
            raise ValueError(f"Given chat_id={_manager.chat_id} already exists. Please provide an unique chat id.")

        # create the folers
        _manager.chat_dirpath.mkdir(parents=True, exist_ok=True)
        
        _dt_now_ts = datetime.timestamp(datetime.now())
        _manager.chatlist[_manager.chat_id] = ChatInfo(
            chat_id=_manager.chat_id,
            created_timestamp=_dt_now_ts,
            last_accessed_timestamp=_dt_now_ts,
        )
        return _manager
    
    @staticmethod
    def from_chat_history(chat_id: str) -> 'ChatManager':
        _manager = ChatManager(chat_id=chat_id)
        _chatlist = _manager.chatlist
        
        if not _chatlist.get(_manager.chat_id, None):
            raise Exception(f"Invalid chat_id: {_manager.chat_id}")
        
        with open(_manager.get_chat_dirpath(_manager.chat_id), 'r') as chat_file:
            _manager._chat = Chat(**json.load(chat_file))
        return _manager
    
    def __init__(self,
        root_dir: Optional[str] = None,
        chat_id: Optional[str] = None,
        chatlist: Optional[Dict[str, ChatInfo]] = None,
        chat: Optional[Chat] = None
    ) -> None:
        self.root_dir = root_dir or self._root_dir

        if chat_id: self._chat_id = chat_id
        if chatlist: self._chat_list = chatlist
        if chat: self._chat = chat
    
    def __del__(self) -> None:
        """Delete the chat folder if it's empty; save it otherwise."""
        try:
            if not list(self.chat_dirpath.iterdir()):
                self.chat_dirpath.rmdir()
            else:
                self.save_chat()
        except Exception as e:
            logger.error(str(e))

    def load_chatlist(self) -> Dict[str, ChatInfo]:
        with open(self._chatlist_filepath, 'r') as chat_list_file:
            return json.load(chat_list_file)
    
    def refresh_chatlist(self) -> Dict[str, ChatInfo]:
        self._chat_list = self.load_chatlist()
        return self._chat_list
    
    def _save_chatlist(self) -> None:
        with open(self._chatlist_filepath, 'w') as chat_list_file:
            json.dump(self._chat_list, chat_list_file, indent=4)
    
    def get_chat_dirpath(self, chat_id: str) -> Path:
        return self.root_dir / self.chats_dir / chat_id / self.chat_filename

    def _save_chat(self, indent: int = 4) -> None:
        with open(self.get_chat_dirpath(self._chat_id), 'w') as chat_file:
            json.dump(self.chat.model_dump(), chat_file, indent=indent)
    
    def save_chat(self) -> None:
        self.chatlist[self.chat_id]["last_accessed_timestamp"] = (
            datetime.timestamp(datetime.now())
        )
        self._save_chat()
        self._save_chatlist()
    
    def delete_current_chat(self) -> None:
        shutil.rmtree(self.chat_dirpath)
    
    def delete_chat(self, chat_id: str) -> None:
        """Raises `FileNotFoundError` if the chat folder does not exist."""
        shutil.rmtree(self.get_chat_dirpath(chat_id))
    
    def add_message(self, message: BaseMessage) -> None:
        self.chat.messages.append(message)
    
    def remove_unlisted_chats(self, excluded_ids: Optional[List[str]] = None) -> None:
        """Remove the chats that are not in the `chatlist.json`.
        
        Args:
            excluded_ids (Optional[List[str]], optional): List of ids to exclude that are not in `chatlist.json`. Defaults to None.
        """
        
        chats_dirpath = self.root_dir / self.chats_dir
        chat_ids = set(chats_dirpath.iterdir())
        excluded_ids = set((excluded_ids or []) + [self.chat_id] + list(self.chatlist.keys()))

        for chat_id in list((chat_ids - excluded_ids) | (excluded_ids - chat_ids)):
            try:
                logger.warning(f"Removing Unlisted Chat: {chat_id}")
                path: Path = chats_dirpath / chat_id
                # Failsafe: what if the path does not exist?
                if path.exists():
                    logger.error(f"Chat: {chat_id} is non-existent!")
                    continue
                shutil.rmtree(path)
            except OSError as e:
                logger.error(str(e))
