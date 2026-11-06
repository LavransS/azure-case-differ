"""Answer questions about TFT patches from the retrieved patch notes.

Each answer is logged to logs/retrieval.jsonl with the chunks it was based on.

Usage:
    python scripts/answers.py "Why was Veigar nerfed in 18.4?"
    python scripts/answers.py "What changed for Kobuko?" --mode vector --top 5
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from azure.search.documents import SearchClient
from dotenv import load_dotenv
from openai import OpenAI

from create_index import require_env
from search import MODES, search_chunks
from upload_chunks import create_openai_client, create_search_client

LOG_PATH = Path("logs/retrieval.jsonl")
DEFAULT_TOP = 8
CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """\
You answer questions about Teamfight Tactics (TFT) patch notes.

Rules:
- Use only the numbered sources in the message. Never use your own knowledge of \
TFT, even if you are confident.
- If the sources don't contain the answer, say that the patch notes you have \
don't cover it. Don't guess. The sources only record changes, so they may not \
explain what a unit or trait does.
- Cite every claim with its source number, like [1].
- A mid-patch update overrides the main notes of the same patch. Among several \
updates, the latest date wins. Among patches, the newest patch wins.
- When a value changed more than once, give the final value and mention the \
earlier one.
- Each set has its own units and traits, so only combine sources from the same \
set when describing one unit's history.
- Be brief, and answer in the same language as the question.\
"""


def format_sources(chunks: list[dict]) -> str:
    """Number the chunks so the model can cite them."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        kind = "mid-patch update" if chunk["is_mid_patch"] else "main patch notes"
        blocks.append(
            f"[{number}] Patch {chunk['patch']}, published {chunk['patch_date'][:10]}, "
            f"{kind}\n{chunk['content']}"
        )
    return "\n\n".join(blocks)


def generate_answer(client: OpenAI, deployment: str, question: str, chunks: list[dict]) -> str:
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Sources:\n\n{format_sources(chunks)}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


def describe_sources(chunks: list[dict], answer: str) -> list[dict]:
    """List what was retrieved, and which of those chunks the answer cited."""
    cited = {int(number) for number in CITATION.findall(answer)}
    return [
        {
            "number": number,
            "cited": number in cited,
            "id": chunk["id"],
            "patch": chunk["patch"],
            "section": chunk["section"],
            "subsection": chunk["subsection"],
            "is_mid_patch": chunk["is_mid_patch"],
            "score": chunk["score"],
            "url": chunk["url"],
        }
        for number, chunk in enumerate(chunks, start=1)
    ]


def answer_question(
    search_client: SearchClient,
    openai_client: OpenAI,
    embedding_deployment: str,
    chat_deployment: str,
    question: str,
    mode: str = "hybrid",
    top: int = DEFAULT_TOP,
) -> dict:
    """Run the full chain: search, generate an answer, and describe the sources."""
    chunks = search_chunks(
        search_client, openai_client, embedding_deployment, question, mode, top
    )
    answer = generate_answer(openai_client, chat_deployment, question, chunks)
    return {
        "question": question,
        "mode": mode,
        "answer": answer,
        "sources": describe_sources(chunks, answer),
    }


def log_result(result: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now(timezone.utc).isoformat(), **result}
    with LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def print_result(result: dict) -> None:
    print(f"\n{result['answer']}\n\nSources ({result['mode']} search):")
    for source in result["sources"]:
        marker = "cited" if source["cited"] else "     "
        heading = " > ".join(filter(None, [source["section"], source["subsection"]]))
        print(f"  [{source['number']}] {marker} {source['patch']} | {heading}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP)
    args = parser.parse_args()

    load_dotenv()
    result = answer_question(
        create_search_client(),
        create_openai_client(),
        require_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        require_env("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        args.question,
        args.mode,
        args.top,
    )
    log_result(result)
    print_result(result)


if __name__ == "__main__":
    main()
