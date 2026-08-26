from typing import List
import logfire

def chunk_text(text : str , chunk_size: int = 1500) -> list[str]:
    """
    Splits the text into chunks of size chunk_size.
    """
    with logfire.span("chunkinh" , text_length = len(text)):
        if not text or not text.strip():
            logfire.info("Empty text provided")
            return []
        
        paragraph = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraph:
            if len(current_chunk) + len(p) < chunk_size:
                current_chunk += p + "\n\n"

            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        valid_chunks = [c for c in chunks if c.strip()]
        logfire.info("Chunking completed", chunk_count=len(chunks))
        return valid_chunks
