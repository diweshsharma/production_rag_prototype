import os 
from dotenv import load_dotenv

load_dotenv()

class config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    QDRANT_URL = os.getenv('QDRANT_CLUSTER_ENDPOINT')
    QDRANT_COLLECTION_NAME = "production_grad_rag"  

    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GROQ_MODEL = os.getenv('llama-3.3-70b-versatile')
    GROQ_FALLBACK_API = os.getenv('GROQ_FALLBACK_API_KEY')
    

   
config = config()
