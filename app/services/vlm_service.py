from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..schemas import ImageContext
from ..config import settings
import base64
from . import model_provider

def generate_image_descriptions(
    images: list[ImageContext]
) -> list[dict]:
    if not settings.GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in settings.")
    
    model = model_provider.get_llm(temperature=0.3)
    
    system_message = SystemMessage(
        content="""You are an expert at analyzing images for a personal journal. 
        Your task is to describe the emotional mood, key subjects, 
        and any significant actions or context in the image. 
        Be descriptive but concise."""
    )
    
    batch_messages = []
    for image in images:
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image.content}"
                    }
                }
            ]
        )
        
        batch_messages.append([system_message, human_message])
    
    descriptions = []
    try:
        batch_responses = model.batch(batch_messages)
        
        for i, response in enumerate(batch_responses):
            descriptions.append({
                "position": images[i].position_after_paragraph,
                "description": response.content
            })
            
    except Exception as e:
        raise Exception(f"An error occurred during VLM batch processing: {e}")

    return descriptions
