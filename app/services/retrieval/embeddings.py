import logfire
import time 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import config
from sentence_transformers import SentenceTransformer

Batch_size = 50
Gemini_DIM = 3072
Fallback_Dim = 768 #all-mpnet-base-v2

active_model = None
model_type: str | None = None

def probe_gemini():
    """Try one embed call to verify Gemini is reachable"""
    try:
        model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",
        google_api_key = config.GEMINI_API_KEY)
        model.embed_query("test")
        logfire.info("gemini embeddings ready")
        return model
    except Exception as e:
        logfire.warning(f"gemini embeddings not ready: {e}")
        return None
    


def load_fall():
    logfire.info(f"Loading fallback SentenceTransformer model(all-mpnet-base-v2 , 768dim)")
    return SentenceTransformer("all-mpnet-base-v2")


def _init():
    global active_model, model_type

    if active_model is not None:
        return

    gemini = probe_gemini()
    if gemini:
        active_model = gemini
        model_type = 'gemini'

    else:
        active_model = load_fall()
        model_type = 'fallback'
    

def get_embed_dim() -> int:
    """return the vector dim for the active model."""
    _init()
    return Gemini_DIM if model_type == 'gemini' else Fallback_Dim


def embed_batch(batch: list[str]) -> list[list[float]]:
    _init()
    if model_type == 'gemini':
        for attempt in range(4):
            try:
                return active_model.embed_documents(batch)
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429" , "rate", "quota" , "resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait= 2 * (2 ** attempt)
                    logfire.warning(
                        f"gemini rate limit hit (attempt {attempt+1}/3) , retrying in {wait}s"
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"gemini embeddings failed after {attempt+1} attempts: {e}")
                    raise
        raise RuntimeError("Gemini rate limit persisted after 4 attemps")

    else:
        return active_model.encode(batch , show_progress_bar = False).tolist()

def embed_query(query:str) -> list[float]:
    _init()
    if model_type == 'gemini':
        return active_model.embed_query(query)
    else:
        return active_model.encode(query).tolist()
        
def embed_text(texts : list[str]) -> list[list[float]]:
    _init()
    all_embeddings : list[list[float]] = []
    for i in range(0 , len(texts) , Batch_size):
        batch = texts[i : i + Batch_size]
        with logfire.span(f"Embedding batch" , model = model_type, start = i , size = len(batch)):
            all_embeddings.extend(embed_batch(batch))

    return all_embeddings


