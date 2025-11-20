from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from app.cloud.storage_client import load_image, pil_image_to_data_url
from ..schemas import ImageContext
from ..config import settings
import base64
from . import model_provider

def generate_image_descriptions(images: list[ImageContext]) -> list[dict]:
    if not images:
        return []

    if not settings.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is required for image description generation.")

    llm = model_provider.get_llm()
    descriptions: list[dict] = []

    for image in images:
        if not image.url.startswith("https://storage.googleapis.com"):
            raise ValueError("Image URL must point to https://storage.googleapis.com")

        pil_image = load_image(image.url)
        data_url = pil_image_to_data_url(
            pil_image,
            image.format,
            getattr(image, "encoding", None),
        )

        prompt = [
            SystemMessage(content="You are an assistant that describes images for emotion analysis."),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe the key emotional cues in this image."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            ),
        ]
        response = llm.invoke(prompt)

        descriptions.append(
            {
                "description": response.content.strip(),
                "position": image.position_after_paragraph,
            }
        )

    return descriptions