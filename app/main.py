import time
from fastapi import FastAPI, Request, Depends, Query, HTTPException, status
from typing import Optional
from langchain_core.messages import HumanMessage

from .dependencies import verify_api_key
from .logutils.logger import get_logs, log_request

from .services import (
    classification_service, 
    embedding_service, 
    illustration_service,
    elaboration_service,
    session_service
)

from .schemas import(
    LogFilters, 
    ClassificationRequest, 
    ClassificationResponse, 
    IllustrationRequest, 
    IllustrationResponse,
    ElaborationChatRequest,
    ElaborationChatResponse,

)

app = FastAPI()
@app.on_event("startup")
async def startup_event():
    embedding_service.initialize_embeddings()

@app.post("/classify", dependencies=[Depends(verify_api_key)])
async def classify(
    request: Request,
    payload: ClassificationRequest
):
    start_time = time.perf_counter()
    
    try:
        result = classification_service.classify_journal(payload)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_request(
            request=request,
            status_code=200,
            latency_ms=latency_ms,
            success=True,
            prediction=result["emotion_classification"]["emotion"],
            confidence=result["emotion_classification"]["similarity"]
        )

        response_data = {
            "emotion_classification": result["emotion_classification"],
            "emotion_tags": result["emotion_tags"],
            "latency_ms": latency_ms
        }
        return ClassificationResponse(**response_data)
    
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_request(request, 500, latency_ms, False, error_message=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {e}"
        )
        
@app.post("/generate-illustration", dependencies=[Depends(verify_api_key)])
async def generate_illustration(
    request: Request,
    payload: IllustrationRequest
):
    start_time = time.perf_counter()
    try:
        illustrable_paragraph, position = illustration_service.identify_illustrable_paragraph(payload.journal_text)

        # 1. Count the total number of paragraphs in the journal
        paragraphs = [p for p in payload.journal_text.split('\n\n') if p.strip()]
        total_paragraphs = len(paragraphs)
        
        final_position = -1
        for i in range(position, 0, -1):
            if i not in payload.filled_paragraph:
                final_position = i
                break
        
        if final_position == -1:
            for i in range(position + 1, total_paragraphs + 1):
                if i not in payload.filled_paragraph:
                    final_position = i
                    break
        
        if final_position == -1:
            final_position = 0 

        visual_essence = illustration_service.extract_visual_essence(illustrable_paragraph)
        
        final_prompt = illustration_service.assemble_illustration_prompt(
            visual_essence=visual_essence,
            style_preference=payload.style_preference,
        )
        
        generated_images = illustration_service.generate_illustration(
            prompt=final_prompt,
            num_images=payload.num_images
        )
        
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_request(request, 200, latency_ms, True)

        return IllustrationResponse(
            images=generated_images,
            prompt=final_prompt,
            position_after_paragraph=final_position,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_request(request, 500, latency_ms, False, error_message=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {e}"
        )
        
@app.post(
    "/elaboration-chat", response_model=ElaborationChatResponse, dependencies=[Depends(verify_api_key)])
async def elaboration_chat(request: ElaborationChatRequest):
    print(request.task)
    
    session = session_service.get_session(request.uuid)
    
    if request.task == "elaborate":
        suggestion = elaboration_service.analyze_journal_for_elaboration(
            journal_text=request.journal_data.text,
            excluded_highlights=session.excluded_highlights,
            chat_history=session.chat_history
        )
        
        if not suggestion:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No more paragraphs to elaborate on or journal is too short."
            )
            
        if suggestion.paragraph_index != -1:
            session.excluded_highlights.add(suggestion.highlight_text)
            
        session.chat_history.add_elaborate_interaction(
            journal_data=request.journal_data,
            suggestion=suggestion
        )
        
        return ElaborationChatResponse(
            uuid=request.uuid,
            elaboration_suggestion=suggestion
        )
        
    elif request.task == "ask":
        if not request.prompt or not request.prompt.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt is required for 'ask' tasks."
            )
        
        assistant_response = elaboration_service.generate_ask_response(
            chat_history=session.chat_history,
            prompt=request.prompt
        )
        
        session.chat_history.add_ask_interaction(
            journal_data=request.journal_data,
            prompt=request.prompt,
            assistant_response=assistant_response
        )
        
        return ElaborationChatResponse(
            uuid=request.uuid,
            assistant_response=assistant_response
        )
        
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid request"
        )


@app.get("/logs", dependencies=[Depends(verify_api_key)])
async def logs(
    request: Request,
    filters: LogFilters
):
    start_time = time.perf_counter()
    
    try:
        
        active_filters = filters.dict(exclude_unset=True)
        logs_data = get_logs(active_filters)
        
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_request(request, 200, latency_ms, True)
        
        return {
            "logs": logs_data,
            "count": len(logs_data),
            "latency_ms": time.perf_counter() - start_time
        }
        
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_request(request, 500, latency_ms, False, error_message=str(e))
        
        return {
            "error": f"Internal server error: {str(e)}"
        }, 500