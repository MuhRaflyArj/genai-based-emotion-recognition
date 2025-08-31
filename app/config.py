import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class Settings:
    API_KEY: str = os.getenv("API_KEY", "your_default_secret_key")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "openai_api_key") 

settings = Settings()