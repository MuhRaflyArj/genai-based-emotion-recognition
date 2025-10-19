from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import vertexai
from vertexai.vision_models import ImageGenerationModel

from ..config import settings

def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in settings.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=temperature,
        google_api_key=settings.GOOGLE_API_KEY
    )

def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in settings.")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GOOGLE_API_KEY
    )
    
def get_imagen_model() -> ImageGenerationModel:
    if not settings.GCP_PROJECT or not settings.GCP_LOCATION:
        raise ValueError("GCP_PROJECT and GCP_LOCATION must be set for image generation.")
    
    vertexai.init(project=settings.GCP_PROJECT, location=settings.GCP_LOCATION)
    
    model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    return model