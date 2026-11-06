"""Create the Azure AI Search index for the patch note chunks.

This only defines the index. Chunks are uploaded in a separate step.

Usage:
    python scripts/create_index.py
    python scripts/create_index.py --recreate    # delete the index and its data first
"""

import argparse
import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    LexicalAnalyzerName,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)
from dotenv import load_dotenv

# Must match the embedding model: text-embedding-3-small returns 1536 numbers.
EMBEDDING_DIMENSIONS = 1536
ALGORITHM_NAME = "hnsw-cosine"
VECTOR_PROFILE_NAME = "default-vector-profile"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing {name}. Add it to the .env file and save it.")
    return value


def build_fields() -> list[SearchField]:
    """Define the index columns and how each one can be used."""
    return [
        SimpleField(name="id", type=SearchFieldDataType.STRING, key=True),
        # Keyword search reads this field. The analyzer treats "buffs" and
        # "buffed" as the same word.
        SearchableField(name="content", analyzer_name=LexicalAnalyzerName.EN_MICROSOFT),
        # Vector search reads this field. It is never shown, so it isn't returned.
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.SINGLE),
            searchable=True,
            retrievable=False,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
        # Text, not a number: as a number, 18.10 would become 18.1.
        SimpleField(name="patch", type=SearchFieldDataType.STRING, filterable=True),
        SimpleField(
            name="patch_date",
            type=SearchFieldDataType.DATE_TIME_OFFSET,
            filterable=True,
            sortable=True,
        ),
        SimpleField(name="category", type=SearchFieldDataType.STRING, filterable=True),
        SimpleField(name="is_mid_patch", type=SearchFieldDataType.BOOLEAN, filterable=True),
        # Returned with each result so the sources can be shown.
        SimpleField(name="title", type=SearchFieldDataType.STRING),
        SimpleField(name="section", type=SearchFieldDataType.STRING),
        SimpleField(name="subsection", type=SearchFieldDataType.STRING),
        SimpleField(name="url", type=SearchFieldDataType.STRING),
        SimpleField(name="source_file", type=SearchFieldDataType.STRING),
    ]


def build_vector_search() -> VectorSearch:
    """Choose how vectors are compared and link that to a named profile."""
    return VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name=ALGORITHM_NAME,
                parameters=HnswParameters(metric=VectorSearchAlgorithmMetric.COSINE),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=ALGORITHM_NAME,
            )
        ],
    )


def build_index(name: str) -> SearchIndex:
    return SearchIndex(name=name, fields=build_fields(), vector_search=build_vector_search())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="delete the existing index and all its data first",
    )
    args = parser.parse_args()

    load_dotenv()
    endpoint = require_env("AZURE_SEARCH_ENDPOINT")
    key = require_env("AZURE_SEARCH_KEY")
    index_name = require_env("AZURE_SEARCH_INDEX")

    client = SearchIndexClient(endpoint, AzureKeyCredential(key))
    if args.recreate:
        client.delete_index(index_name)
        print(f"Deleted index '{index_name}'")

    index = client.create_or_update_index(build_index(index_name))
    print(f"Index '{index.name}' is ready with {len(index.fields)} fields")


if __name__ == "__main__":
    main()
