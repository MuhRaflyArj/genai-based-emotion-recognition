import json
from typing import List
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from ..schemas import JournalData, ElaborationSuggestion

class StructuredJournalHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    
    def _format_message_content(self, data: dict) -> str:
        return json.dumps(data, indent=2)
    
    def add_elaborate_interaction(
        self,
        journal_data: JournalData,
        suggestion: ElaborationSuggestion
    ):
        human_content = self._format_message_content({
            "task": "elaborate",
            "journal_text": journal_data.text
        })
        self.add_user_message(human_content)
        
        ai_content = self._format_message_content({
            "strategy_used": suggestion.strategy_used,
            "suggestion_text": suggestion.suggestion_text,
            "highlight_text": suggestion.highlight_text
        })
        self.add_ai_message(ai_content)
        
    def add_ask_interaction(
        self,
        journal_data: JournalData,
        prompt: str,
        assistant_response: str
    ):
        human_content = self._format_message_content({
            "task": "ask",
            "journal_text": journal_data.text,
            "prompt": prompt
        })
        self.add_user_message(human_content)
        
        ai_content = self._format_message_content({
            "assistant_response": assistant_response
        })
        self.add_ai_message(ai_content)
        
    def add_messages(
        self,
        messages: List[BaseMessage]
    ) -> None:
        self.messages.extend(messages)
        
    def clear(self) -> None:
        self.messages = []