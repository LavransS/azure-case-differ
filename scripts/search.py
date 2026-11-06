"""Search the patch notes with keyword, vector or hybrid search.

Usage:
    python scripts/search.py "Why was Veigar nerfed in 18.4?"
    python scripts/search.py "Why was Veigar nerfed?" --mode hybrid --top 3
"""

import argparse

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import OpenAI

from create_index import require_env
from upload_chunks import create_openai_client, create_search_client, embed_texts

MODES = ("keyword", "vector", "hybrid")
RESULT_FIELDS = [
    "id",
    "patch",
    "patch_date",
    "title",
    "section",
    "subsection",
    "category",
    "is_mid_patch",
    "content",
    "url",
]
# Vector search looks at this many neighbours before results are cut to `top`.
# Extra candidates give hybrid search more to combine.
VECTOR_CANDIDATES = 50
# Cosmetic lists are full of champion names, which crowd out balance changes.
DEFAULT_FILTER = "category ne 'cosmetics'"


def search_chunks(
    search_client: SearchClient,
    openai_client: OpenAI,
    deployment: str,
    question: str,
    mode: str = "hybrid",
    top: int = 5,
    filter_expression: str | None = DEFAULT_FILTER,
) -> list[dict]:
    """Return the best matching chunks, best first."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {MODES}.")

    vector_queries = None
    if mode in ("vector", "hybrid"):
        vector = embed_texts(openai_client, deployment, [question])[0]
        vector_queries = [
            VectorizedQuery(
                vector=vector,
                k_nearest_neighbors=VECTOR_CANDIDATES,
                fields="content_vector",
            )
        ]

    results = search_client.search(
        search_text=None if mode == "vector" else question,
        vector_queries=vector_queries,
        filter=filter_expression,
        select=RESULT_FIELDS,
        top=top,
    )
    return [{**result, "score": result["@search.score"]} for result in results]


def print_results(mode: str, results: list[dict]) -> None:
    print(f"\n{mode.upper()}")
    if not results:
        print("  No results")
    for rank, result in enumerate(results, start=1):
        heading = " > ".join(filter(None, [result["section"], result["subsection"]]))
        print(f"  {rank}. [{result['score']:.3f}] {result['patch']} | {heading}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, help="run one mode instead of all three")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    load_dotenv()
    deployment = require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    openai_client = create_openai_client()
    search_client = create_search_client()

    print(f"Question: {args.question}")
    for mode in [args.mode] if args.mode else MODES:
        results = search_chunks(
            search_client, openai_client, deployment, args.question, mode, args.top
        )
        print_results(mode, results)


if __name__ == "__main__":
    main()
