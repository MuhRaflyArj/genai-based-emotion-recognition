from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from typing import Dict, Set
from pydantic import BaseModel, Field

class UserSession(BaseModel):
    chat_memory: BaseChatMemory
    suggested_paragraph_indicies: Set[int] = Field(default_factory=set)
    
    class Config:
        arbitrary_types_allowed = True
        
session_store: Dict[str, UserSession] = {}

def create_session_memory(uuid: str) -> BaseChatMemory:
    if uuid not in session_store:
        chat_memory_instance = ConversationBufferWindowMemory(
            k=5,
            memory_key="history",
            return_messages=True
        )
        
        session_store[uuid] = UserSession(chat_memory=chat_memory_instance)
        
    return session_store[uuid]