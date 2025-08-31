from langchain_openai import OpenAIEmbeddings
from ..config import settings
from scipy.spatial.distance import cosine

emotion_categories = {
    # --- Positive Emotions ---
    "a feeling of joy and happiness": 
        "This text expresses a strong positive emotion like joy, happiness, delight, elation, or triumph. It often relates to moments of success, fun, connection with loved ones, or pure, vibrant contentment.",

    "a moment of peacefulness and calm": 
        "This describes a state of peacefulness, calm, serenity, relaxation, or deep contentment. It reflects a low-energy, positive state, free from stress, turmoil, or anxiety.",

    "an experience of gratitude and appreciation": 
        "This entry expresses feelings of gratitude, appreciation, or thankfulness. The author is actively acknowledging the good things in their life, whether it's people, experiences, or simple pleasures.",

    "a feeling of excitement and anticipation": 
        "This text conveys a high-energy, positive feeling of excitement, anticipation, or hopefulness. It is often associated with looking forward to a future event, a new opportunity, or a positive outcome.",

    # --- Negative Emotions ---
    "a sense of sadness and grief": 
        "This entry describes feelings of sadness, grief, disappointment, hurt, or loneliness. It is often associated with loss, failure, bad news, or a difficult emotional experience.",

    "an expression of anger and frustration": 
        "This text conveys feelings of anger, frustration, irritation, annoyance, or being upset. This emotion is often a reaction to a perceived injustice, an obstacle, a conflict, or a violation of personal boundaries.",

    "a feeling of anxiety and fear": 
        "This entry expresses feelings of anxiety, fear, worry, stress, nervousness, or being overwhelmed. It is often related to uncertainty about the future, a perceived threat, or high-pressure situations.",

    # --- Complex/Reflective Emotions ---
    "a memory filled with nostalgia": 
        "This text has a nostalgic or bittersweet tone, reflecting on the past. It often mixes feelings of warmth and happiness for a memory with a sense of longing or sadness for a time that is now gone.",

    "a story of personal growth and resilience": 
        "This is a narrative of personal growth, resilience, learning, or overcoming a challenge. The text describes a process of development or finding strength through adversity, rather than just a single, static emotion."
}

context_tags_map = {
    "Family": "This text describes feelings or events related to family members and relatives.",
    "Romance & Love": "This entry is about a romantic partner, dating, love life, or feelings of deep affection.",
    "Friendship & Social": "This text discusses friends, social events, community, or a sense of belonging.",
    "Work & Career": "This entry is about a job, career, professional life, or academic studies.",
    "Personal Growth": "This text is about self-improvement, personal insights, learning new skills, or self-development.",
    "Health & Wellness": "This entry describes experiences related to physical health, fitness, diet, sleep, or the body.",
    "Hobbies & Creativity": "This entry is about hobbies, passions, creative pursuits, art, music, or leisure time.",
    "Finances & Money": "This text discusses money, finances, budgeting, or material possessions.",
    "Spirituality & Meaning": "This entry reflects on spirituality, personal beliefs, religion, or a search for meaning and purpose.",
    "Travel & Adventure": "This text is about traveling, exploring new places, adventures, or new experiences.",
    "Milestones & Events": "This entry describes a major life event, a special occasion, a celebration, or a significant milestone.",
    "Challenges & Obstacles": "This text is about dealing with a challenge, a problem, a failure, or a difficult situation.",
    "Achievements & Success": "This entry celebrates a success, a personal accomplishment, a win, or good news.",
    "Daily Life & Routines": "This text describes everyday life, daily routines, chores, or simple, mundane moments.",
    "Reflections & Plans": "This entry is about reflecting on the past, nostalgia, memories, or making plans for the future."
}

embedding_store = {
    "classifications": {},
    "tags": {}
}

def get_embedding_model():
    if not settings.OPENAI_API_KEY:
        raise ValueError("OpenAI API key is invalid.")
    
    model = OpenAIEmbeddings(
        model="text-embedding-3-large", 
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    return model

def initialize_embeddings():
    model = get_embedding_model()
    
    emotion_descriptions = list(emotion_categories.values())
    tag_descriptions = list(context_tags_map.values())
    
    all_texts_to_embed = emotion_descriptions + tag_descriptions
    
    all_embeddings = model.embed_documents(all_texts_to_embed)
    
    num_emotion_categories = len(emotion_categories)
    
    emotion_labels = list(emotion_categories.keys())
    for i, label in enumerate(emotion_labels):
        embedding_store['classifications'][label] = all_embeddings[i]
        
    tag_embeddings = all_embeddings[num_emotion_categories:]
    tag_labels = list(context_tags_map.keys())
    for i, label in enumerate(tag_labels):
        embedding_store["tags"][label] = tag_embeddings[i]

def embed_document(text: str) -> list[float]:
    model = get_embedding_model()
    return model.embed_query(text)

def calculate_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    return 1 - cosine(vec1, vec2)