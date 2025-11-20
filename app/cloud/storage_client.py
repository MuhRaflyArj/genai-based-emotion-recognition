from __future__ import annotations

import base64
import io
from urllib.parse import urlparse
import uuid

from google.cloud import storage
from PIL import Image

from app.config import settings

credentials_path = settings.GOOGLE_APPLICATION_CREDENTIALS
storage_client = storage.Client.from_service_account_json(credentials_path)

def get_bucket_name() -> str:
    bucket_name = settings.BUCKET_NAME
    if not bucket_name:
        raise RuntimeError("BUCKET_NAME must be set in the environment.")
    return bucket_name

def generate_hashed_filename(extension: str) -> str:
    ext = extension.lower().lstrip(".") or "png"
    unique_id = uuid.uuid4().hex
    return f"{unique_id}.{ext}"


def build_illustration_blob_path(user_id: str, journal_id: str, filename: str) -> str:
    return f"uploads/videos/{user_id}/{journal_id}/illustrations/image_uploads/{filename}"


def upload_bytes_to_bucket(data: bytes, blob_path: str, content_type: str) -> str:
    bucket = storage_client.bucket(get_bucket_name())
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)
    return f"https://storage.googleapis.com/{bucket.name}/{blob_path}"

def parse_gcs_https_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "storage.googleapis.com":
        raise ValueError("URL must start with https://storage.googleapis.com")
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) != 2:
        raise ValueError("Invalid GCS URL format; expected /<bucket>/<object>")
    return parts[0], parts[1]


def download_blob_bytes(url: str) -> bytes:
    bucket_name, blob_name = parse_gcs_https_url(url)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()


def load_image(url: str) -> Image.Image:
    image_bytes = download_blob_bytes(url)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def determine_mime_type(format_hint: str | None, encoding_hint: str | None) -> str:
    if encoding_hint:
        return encoding_hint
    if format_hint:
        ext = format_hint.lower().lstrip(".")
        if ext in {"jpg", "jpeg"}:
            return "image/jpeg"
        if ext == "png":
            return "image/png"
        if ext == "webp":
            return "image/webp"
    return "image/jpeg"


def pil_image_to_data_url(
    pil_image: Image.Image,
    format_hint: str | None,
    encoding_hint: str | None,
) -> str:
    buffer = io.BytesIO()
    fmt = (format_hint or "jpeg").upper()
    if fmt == "JPG":
        fmt = "JPEG"
    pil_image.save(buffer, format=fmt)
    mime_type = determine_mime_type(format_hint, encoding_hint)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"