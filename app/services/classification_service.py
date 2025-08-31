from ..schemas import ClassificationRequest, EntryData
from . import vlm_service
from . import embedding_service

def classify_journal(
    payload: ClassificationRequest
) -> dict:
    image_descriptions = []
    if payload.media_context and payload.media_context.images:
        image_descriptions = vlm_service.generate_image_descriptions(
            payload.media_context.images
        )
        
    super_document = construct_super_document(
        entry_data = payload.entry_data,
        video_emotion = payload.media_context.video_emotion if payload.media_context else None,
        video_confidence = payload.media_context.video_confidence if payload.media_context else None,    
        image_descriptions = image_descriptions
    )

    doc_embedding = embedding_service.embed_document(super_document)
    
    classification_results = []
    for label, label_embedding in embedding_service.embedding_store["classifications"].items():
        similarity = embedding_service.calculate_cosine_similarity(doc_embedding, label_embedding)
        classification_results.append({"emotion": label, "similarity": similarity})
    
    best_classification = max(classification_results, key=lambda x: x['similarity'])
    
    all_tags = []
    for label, label_embedding in embedding_service.embedding_store["tags"].items():
        similarity = embedding_service.calculate_cosine_similarity(doc_embedding, label_embedding)
        all_tags.append({"tags": label, "similarity": similarity})
        
    if not all_tags:
        return {
            "emotion_classification": best_classification,
            "emotion_tags": []
        }
        
    sorted_tags = sorted(all_tags, key=lambda x: x['similarity'], reverse=True)
    top_tags = [sorted_tags[0]]
    best_tag_score = sorted_tags[0]['similarity']  
    threshold = best_tag_score * 0.80
    
    candidate_tags = []
    for tag in sorted_tags[1:]:
        if tag['similarity'] >= threshold:
            candidate_tags.append(tag)
            
    top_tags.extend(candidate_tags[:2])
    
    return {
        "emotion_classification": best_classification,
        "emotion_tags": top_tags
    }


def construct_super_document(
    entry_data: EntryData,
    video_emotion: str,
    video_confidence: float,
    image_descriptions: list[dict]
) -> str:
    sorted_images = sorted(image_descriptions, key=lambda x: x['position'])
    paragraphs = entry_data.text.split('\n\n')
    
    content_parts = []
    content_parts.append(f'Title: {entry_data.title}')
    
    if video_emotion:
        content_parts.append(f"Context from video: The emotion was {video_emotion}.")
        content_parts.append(f"AI Confidence score from the video: {video_confidence}")
        
    img_idx = 0
    for i, paragraph in enumerate(paragraphs):
        content_parts.append(paragraph)
        while img_idx < len(sorted_images) and sorted_images[img_idx]['position'] == i:
            img_desc = sorted_images[img_idx]['description']
            content_parts.append(f"[Image Description: {img_desc}]")
            img_idx += 1
            
    while img_idx < len(sorted_images):
        img_desc = sorted_images[img_idx]['description']
        content_parts.append(f"[Image Description: {img_desc}]")
        img_idx += 1
        
    return "\n".join(content_parts)