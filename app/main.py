import time
from fastapi import FastAPI, Request, Depends, Query, HTTPException, status
from typing import Optional

from .dependencies import verify_api_key
from .logutils.logger import get_logs, log_request

from .services import (
    classification_service, 
    embedding_service, 
    illustration_service
)

from .schemas import(
    LogFilters, 
    ClassificationRequest, 
    ClassificationResponse, 
    IllustrationRequest, 
    IllustrationResponse
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
            position_after_paragraph=position,
            latency_ms=latency_ms
        )
        
    except Exception as e:
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        log_request(request, 500, latency_ms, False, error_message=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Server Error: {e}"
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