import json
import base64
from . import model_provider
from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from ..config import settings
from . import model_provider

from app.cloud.storage_client import (
    build_illustration_blob_path,
    generate_hashed_filename,
    upload_bytes_to_bucket,
)

class VisualEssence(BaseModel):
    """A list of concise descriptive phrases representing the visual essence of a text."""
    visual_elements: List[str] = Field(
        ..., 
        description="A list of strings, where each string is a concise descriptive phrase of a visual element (subject, object, setting, action) from the text."
    )

def identify_illustrable_paragraph(journal_text: str) -> str:
    
    paragraphs = [p.strip() for p in journal_text.split('\n\n') if p.strip()]
    if not paragraphs:
        raise ValueError("Journal text is empty or contains no valid paragraphs.")
    
    numbered_journal_text = ""
    for i, p in enumerate(paragraphs):
        numbered_journal_text += f"Paragraph {i + 1}:\n{p}\n\n"

    llm = model_provider.get_llm(temperature=0.0)
    system_message = SystemMessage(
        content=f"""You are an expert in visual storytelling. Your task is to analyze the following journal entry, which is split into numbered paragraphs. 
        Identify the single paragraph that is the most visually descriptive and suitable for creating an illustration. 
        Consider paragraphs with concrete nouns, actions, and sensory details.
        
        Your response must be ONLY the number of the chosen paragraph (e.g., '2'). Do not include any other text, punctuation, or explanation.
        There are {len(paragraphs)} paragraphs in total."""
    )
    human_message = HumanMessage(content=numbered_journal_text)
    
    try:
        response = llm.invoke([system_message, human_message])
        paragraph_number = int(response.content.strip())

        if not (1 <= paragraph_number <= len(paragraphs)):
            raise ValueError(f"LLM returned an invalid paragraph number: {paragraph_number}")

        position = paragraph_number
        chosen_paragraph = paragraphs[position - 1]
        
        return chosen_paragraph, position

    except (ValueError, TypeError) as e:
        raise Exception(f"Failed to parse a valid paragraph number from LLM response: {e}")
    except Exception as e:
        raise Exception(f"Failed to identify illustrable paragraph: {e}")


def extract_visual_essence(paragraph: str) -> list[str]:

    llm = model_provider.get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(VisualEssence)
    
    system_message = SystemMessage(
        content="""You are an expert in extracting visual details from text for an art generation model.
        From the given paragraph, identify the key visual elements (subjects, objects, setting, actions).

        **IMPORTANT SAFETY RULE:** Your primary goal is to interpret the user's text in a way that is safe for an AI image generator.
        - **DO NOT** extract any elements that depict or imply self-harm, violence, gore, explicit adult content, or hate symbols.
        - If the text contains sensitive themes, rephrase them into abstract or symbolic representations. For example, instead of "a bloody knife," extract "a crimson object casting a long shadow." Instead of a violent act, describe the emotional aftermath, like "a sense of turmoil represented by stormy clouds."
        - Focus on creating a visually rich and emotionally resonant scene that is artistic and G-rated.

        Return these safe and rephrased elements as a JSON array of strings. Each string should be a concise descriptive phrase.
        Example output: ["a person sitting on a park bench", "autumn leaves falling", "a red scarf", "a distant city skyline"]
        Return ONLY the JSON array."""
    )

    human_message = HumanMessage(content=paragraph)
    
    try:
        response = structured_llm.invoke([system_message, human_message])
        return response.visual_elements
        
    except (json.JSONDecodeError, ValueError, Exception) as e:
        raise Exception(f"Failed to extract visual essence: {e}")

def assemble_illustration_prompt(
    visual_essence: list[str], 
    style_preference: str
) -> str:
    essence_string = ", ".join(visual_essence)
    
    prompt = (
        f"Create a digital illustration in a '{style_preference}' style. "
        f"The scene must feature: {essence_string}. "
        "Focus on a clear composition that tells a story. The overall tone should be artistic and evocative."
    )
    
    return prompt

def generate_illustration(
    prompt: str,
    num_images: int,
    user_id: str,
    journal_id: str,
) -> list[str]:
    model = model_provider.get_imagen_model()
    response = model.generate_images(
        prompt=prompt,
        number_of_images=num_images,
    )

    generated_items = response.images if hasattr(response, "images") else response
    
    if not generated_items:
        raise RuntimeError("Image generation returned no images to upload.")

    uploaded_urls: list[str] = []

    for generated in generated_items:
        image_bytes = getattr(generated, "_image_bytes", None)

        mime_type = getattr(generated, "mime_type", "image/png")
        extension = "png" if "png" in mime_type else "jpg"

        filename = generate_hashed_filename(extension)
        blob_path = build_illustration_blob_path(user_id, journal_id, filename)
        
        public_url = upload_bytes_to_bucket(image_bytes, blob_path, mime_type)

        uploaded_urls.append(public_url)

    if not uploaded_urls:
        raise RuntimeError("No valid images were produced for upload.")

    return uploaded_urls