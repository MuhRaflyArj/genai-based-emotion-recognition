from langchain.memory import ConversationBufferWindowMemory
from langchain.memory.chat_memory import BaseChatMemory
from typing import Dict

session_store: Dict[str, BaseChatMemory] = {}

def create_session_memory(uuid: str) -> BaseChatMemory:
    if uuid not in session_store:
        session_store[uuid] = ConversationBufferWindowMemory(
            k=10,
            memory_key="history",
            return_messages=True
        )
        
    return session_store[uuid]