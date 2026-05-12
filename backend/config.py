from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    UPLOAD_DIR = DATA_DIR / "uploads"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    # Create directories
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Neo4j Configuration (optional for testing)
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    
    # OpenAI Configuration (optional for testing)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    USE_LLM = bool(OPENAI_API_KEY)
    
    # File settings
    MAX_FILE_SIZE = 10485760  # 10MB
    ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.docx', '.md'}

config = Config()