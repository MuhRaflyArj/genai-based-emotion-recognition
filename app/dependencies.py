from fastapi import Header, HTTPException, status, Request
from typing import Optional

from .logutils.logger import log_request
from .config import settings

async def verify_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key")
):
    if not x_api_key or x_api_key != settings.API_KEY:
        log_request(request, 401, 0, False, error_message="Invalid API Key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )