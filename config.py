import os
from dotenv import load_dotenv

# Load environment variables from .env file (never commit .env!)
load_dotenv()

# Access secrets safely via environment variables
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DEBUG = os.getenv("DEBUG", "False").lower() == "true"
