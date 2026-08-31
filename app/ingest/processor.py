import logfire
import os
import sys
import uuid
import json

from app.config import config

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.ingest.loader.pdf import parse_pdf
from app.ingest.loader.html import parse_html
from app.ingest.loader.office import parse_office
from app.ingest.loader.text import parse_text

from app.services.retrieval.embeddings import embed_text, get_embed_dim
from app.ingest.chunking.splitter import chunk_text

logfire.configure(service_name="enterprise-ingestion-service")

processed_data_dir = "processed_data"

qdrant_client = QdrantClient(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY
)


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/<filename>.json"""
    folder = os.path.join(processed_data_dir, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def process_file(file_path: str, filename: str, source_type: str):
    """Parse -> chunk -> save locally -> embed -> index in Qdrant"""

    with logfire.span("processing file", file=filename, source=source_type):
        try:
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext in ("docx", "pptx"):
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"no text extracted from: {filename}")
                return

            chunks = chunk_text(full_text)
            if not chunks:
                return

            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data -> {local_path}")

            with logfire.span("vectorizing & Indexing"):
                embeddings = embed_text(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for chunk, emb in zip(chunks, embeddings)
                ]

                qdrant_client.upsert(
                    collection_name=config.QDRANT_COLLECTION_NAME,
                    points=points,
                )

                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}")

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        for filename in files:
            process_file(os.path.join(dir_path, filename), filename, source_type)


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion."""
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(config.QDRANT_COLLECTION_NAME):
                    qdrant_client.delete_collection(config.QDRANT_COLLECTION_NAME)
                    logfire.info(f"Collection '{config.QDRANT_COLLECTION_NAME}' deleted.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(config.QDRANT_COLLECTION_NAME):
            dim = get_embed_dim()
            qdrant_client.create_collection(
                collection_name=config.QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{config.QDRANT_COLLECTION_NAME}' ({dim}-dim, Cosine)."
            )

    # Route to sub-folders or treat the whole dir as one source
    subdirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not subdirs:
        if explicit_source_type:
            source_type = explicit_source_type
        else:
            base_name = os.path.basename(os.path.normpath(base_dir)).lower()
            source_type = (
                "true" if "true" in base_name
                else "noisy" if "noisy" in base_name
                else "general"
            )
        logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
        process_directory(base_dir, source_type)
    else:
        for subdir in subdirs:
            source_type = (
                "true" if "true" in subdir.lower()
                else "noisy" if "noisy" in subdir.lower()
                else subdir
            )
            process_directory(os.path.join(base_dir, subdir), source_type)


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingest.processor DATA --wipe
    #   python -m app.ingest.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    run_universal_ingestion(target_dir, explicit_type, wipe=wipe_requested)