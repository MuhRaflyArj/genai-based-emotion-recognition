from pydantic import BaseModel
from typing import Optional, List

class LogFilters(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    status_code: Optional[int] = None
    client_id: Optional[str] = None
    success: Optional[bool] = None
    
class ImageContext(BaseModel):
    content: str
    format: str
    position_after_paragraph: int
    
class MediaContext(BaseModel):
    video_emotion: Optional[str] = None
    video_confidence: Optional[float] = None
    images: Optional[List[ImageContext]] = None
    
class EntryData(BaseModel):
    title: str
    text: str
    
class ClassificationRequest(BaseModel):
    entry_data: EntryData
    media_context: Optional[MediaContext] = None
    
class EmotionClassification(BaseModel):
    emotion: str
    similarity: float
    
class EmotionTag(BaseModel):
    tags: str
    similarity: float
    
class ClassificationResponse(BaseModel):
    emotion_classification: EmotionClassification
    emotion_tags: List[EmotionTag]
    latency_ms: int
    
class ClassificationResponse(BaseModel):
    emotion_classification: EmotionClassification
    emotion_tags: List[EmotionTag]
    latency_ms: int

class IllustrationRequest(BaseModel):
    journal_text: str
    style_preference: str = "digital painting"
    num_images: int = 1

class IllustrationResponse(BaseModel):
    images: List[str] # List of base64 encoded images
    prompt: str
    position_after_paragraph: int
    latency_ms: int
    
class ElaborationSuggestion(BaseModel):
    strategy_used: str
    suggestion_text: str
    highlight_text: str

class JournalData(BaseModel):
    text: str
    user_images: Optional[List[str]] = None

class ElaborationChatRequest(BaseModel):
    uuid: str
    journal_data: Optional[JournalData] = None
    user_chat_input: Optional[str] = None
    
class ElaborationChatResponse(BaseModel):
    uuid: str
    elaboration_suggestion: Optional[ElaborationSuggestion] = None
    assistant_response: Optional[str] = None
    is_final_message: bool
    