from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..schemas import ImageContext
from ..config import settings
import base64

def generate_image_descriptions(
    images: list[ImageContext]
) -> list[dict]:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OpenAI API key is invalid.")
    
    model = ChatOpenAI(
        model='gpt-4o-mini',
        openai_api_key = settings.OPENAI_API_KEY,
        temperature=0.35
    )
    
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
                        "url": f"data:image/{image.format};base64,{image.content}"
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
