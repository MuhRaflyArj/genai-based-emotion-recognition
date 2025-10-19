import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Settings:
    API_KEY: str = os.getenv("API_KEY", "your_default_secret_key")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "openai_api_key") 
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "google_api_key")
    GCP_PROJECT: str = os.getenv("GCP_PROJECT", "your_gcp_project")
    GCP_LOCATION: str = os.getenv("GCP_LOCATION", "your_gcp_location")

settings = Settings()