from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ..config import settings

def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OpenAI API key is invalid.")
    return ChatOpenAI(
        model='gpt-4o-mini',
        openai_api_key=settings.OPENAI_API_KEY,
        temperature=temperature
    )

def get_embedding_model() -> OpenAIEmbeddings:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OpenAI API key is invalid.")
    
    return OpenAIEmbeddings(
        model="text-embedding-3-large", 
        openai_api_key=settings.OPENAI_API_KEY
    )
    
def get_dalle_client():
    if not settings.OPENAI_API_KEY:
        raise ValueError("OpenAI API key is invalid.")
    
    return OpenAI(api_key=settings.OPENAI_API_KEY)