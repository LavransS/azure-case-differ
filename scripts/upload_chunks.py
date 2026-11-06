"""Embed the patch note chunks and upload them to Azure AI Search.

Safe to run again: each chunk replaces the document with the same id.

Usage:
    python scripts/upload_chunks.py
"""

import json
import os
import time
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv
from openai import OpenAI

from create_index import EMBEDDING_DIMENSIONS, require_env

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
BATCH_SIZE = 50
# Azure can take a moment before new documents appear in the count.
COUNT_DELAY_SECONDS = 5


def load_chunks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def create_openai_client() -> OpenAI:
    endpoint = require_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    return OpenAI(
        base_url=f"{endpoint}/openai/v1/",
        api_key=require_env("AZURE_OPENAI_KEY"),
        max_retries=5,
    )


def create_search_client() -> SearchClient:
    return SearchClient(
        require_env("AZURE_SEARCH_ENDPOINT"),
        require_env("AZURE_SEARCH_INDEX"),
        AzureKeyCredential(require_env("AZURE_SEARCH_KEY")),
    )


def embed_texts(client: OpenAI, deployment: str, texts: list[str]) -> list[list[float]]:
    """Return one embedding per text, in the same order."""
    response = client.embeddings.create(model=deployment, input=texts)
    vectors = [item.embedding for item in response.data]
    if len(vectors[0]) != EMBEDDING_DIMENSIONS:
        raise SystemExit(
            f"The model returned {len(vectors[0])} numbers per text, but the index "
            f"expects {EMBEDDING_DIMENSIONS}."
        )
    return vectors


def main() -> None:
    load_dotenv()
    deployment = require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    openai_client = create_openai_client()
    search_client = create_search_client()

    chunks = load_chunks(CHUNKS_PATH)
    failed_ids = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        vectors = embed_texts(openai_client, deployment, [c["content"] for c in batch])
        documents = [
            {**chunk, "content_vector": vector}
            for chunk, vector in zip(batch, vectors, strict=True)
        ]
        results = search_client.upload_documents(documents=documents)
        failed_ids.extend(result.key for result in results if not result.succeeded)
        print(f"Uploaded {start + len(batch)} of {len(chunks)} chunks")

    if failed_ids:
        raise SystemExit(f"{len(failed_ids)} chunks failed: {failed_ids}")

    time.sleep(COUNT_DELAY_SECONDS)
    print(f"Index now holds {search_client.get_document_count()} documents")


if __name__ == "__main__":
    main()
