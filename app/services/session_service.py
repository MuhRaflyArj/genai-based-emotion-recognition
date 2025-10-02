from typing import Dict, Set
from pydantic import BaseModel, Field

from .memory_service import StructuredJournalHistory

class UserSession(BaseModel):
    chat_history: StructuredJournalHistory = Field(default_factory=StructuredJournalHistory)
    excluded_highlights: Set[str] = Field(default_factory=set)

    class Config:
        arbitrary_types_allowed = True
        
SESSIONS: Dict[str, UserSession] = {}

def get_session(uuid: str) -> UserSession:
    if uuid not in SESSIONS:
        SESSIONS[uuid] = UserSession()
    return SESSIONS[uuid]