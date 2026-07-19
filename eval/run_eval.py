import httpx
import os
from pathlib import Path
import json
import re
import asyncio

from backend.services.retriever import retrieve, SourceChunk
from backend.services.prompt_builder import build_prompt
from backend.services.llm_client import generate_answer
from backend.dependencies import (
    get_openai_client,
    get_chroma_collection,
    initialize_clients,
)
from backend.core.config import settings

API_BASE_URL = os.getenv("RAGNAR_API_URL", "http://localhost:8000")
DOCUMENTS_DIR = Path(__file__).resolve().parent / "documents"


def ingest_documents():
    if not DOCUMENTS_DIR.exists():
        print(f"Documents directory {DOCUMENTS_DIR} does not exist.")
        return

    for file_path in DOCUMENTS_DIR.iterdir():
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() not in {".pdf", ".txt"}:
            continue

        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f)}
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/ingest/?force=true",
                    files=files,
                    timeout=120.0,
                )
                response.raise_for_status()
                print(f"Successfully ingested {file_path.name}")
            except httpx.HTTPStatusError as e:
                print(
                    f"Failed to ingest {file_path.name}: "
                    f"HTTP {e.response.status_code} - {e.response.text}"
                )
            except httpx.RequestError as e:
                print(f"Failed to ingest {file_path.name}: {e}")


def load_set() -> list[dict[str, str]]:
    eval_set: list[dict[str, str]] = []
    set_path = Path(__file__).resolve().parent / "eval_set.json"
    if not set_path.exists():
        print(f"Evaluation set file {set_path} does not exist.")
        return []

    with open(set_path, "r", encoding="utf-8") as f:
        try:
            eval_set = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Failed to load evaluation set: {e}")
            return []

    return eval_set


def parse_judge_score(raw: str) -> int | None:
    try:
        score = int(raw.strip())
    except ValueError:
        match = re.search(r"\b([1-5])\b", raw)
        if match:
            score = int(match.group(1))
        else:
            print(f"Unparseable: {raw!r}")
            return None

    if score < 1:
        return 1
    if score > 5:
        return 5
    return score


def token_overlap(text_a: str, text_b: str) -> float:
    tokens_a = set(re.sub(r"[^\w\s]", "", text_a.lower()).split())
    tokens_b = set(re.sub(r"[^\w\s]", "", text_b.lower()).split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


async def main():
    ingest_documents()
    await initialize_clients()
    collection = get_chroma_collection()
    client = get_openai_client()

    eval_set = load_set()
    print(f"Loaded {len(eval_set)} evaluation items.\n")

    results: list[dict] = []
    ratings: list[int] = []
    unparseable_count = 0
    recall_hits = 0
    grounding_hits = 0
    grounding_total = 0

    for item in eval_set:
        question = item.get("question", "")
        expected_answer = item.get("expected_answer", "")
        relevant_chunk_text = item.get("relevant_chunk_text", "")
        source_file = item.get("source_file", "")

        retrieved_chunks: list[SourceChunk] = await retrieve(
            query=question,
            collection=collection,
            client=client,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            top_k=settings.top_k,
            min_similarity=settings.min_similarity,
        )

        if not relevant_chunk_text:
            # Unanswerable question: check grounding (should return 0 chunks)
            grounding_total += 1
            grounded = all(
                c.similarity_score < settings.grounding_threshold
                for c in retrieved_chunks
            )
            if grounded:
                grounding_hits += 1
            results.append(
                {
                    "question": question,
                    "recall": grounded,
                    "score": None,
                    "grounding_check": True,
                }
            )
            continue

        recall = any(
            token_overlap(relevant_chunk_text, chunk.text) >= 0.5
            for chunk in retrieved_chunks
        )
        if recall:
            recall_hits += 1

        if not retrieved_chunks:
            results.append(
                {
                    "question": question,
                    "recall": recall,
                    "score": None,
                    "grounding_check": False,
                }
            )
            continue

        messages = build_prompt(
            question=question,
            chunks=retrieved_chunks,
            max_context_tokens=settings.max_context_tokens,
        )
        generated_answer = await generate_answer(
            messages=messages,
            client=client,
            model=settings.chat_model,
        )

        judge_prompt = (
            f"Question: {question}\n"
            f"Expected Answer: {expected_answer}\n"
            f"Generated Answer: {generated_answer.answer}\n"
            f"Rate the quality of the generated answer on a scale of 1–5 "
            f"where 5 is perfect. Respond with only the integer score."
        )
        judge_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": judge_prompt}],
            model=settings.judge_model,
            temperature=0,
        )
        raw_judge = judge_response.choices[0].message.content
        score = parse_judge_score(raw_judge)

        if score is None:
            unparseable_count += 1
        else:
            ratings.append(score)

        results.append(
            {
                "question": question,
                "recall": recall,
                "score": score,
                "grounding_check": False,
            }
        )

    print(f"{'Question':<45} {'Recall':>6} {'Score':>5}")
    print("-" * 58)
    for row in results:
        q_short = (
            row["question"][:42] + "..."
            if len(row["question"]) > 45
            else row["question"]
        )
        recall = "✓" if row["recall"] else "✗"
        if row["grounding_check"]:
            score_str = "GND" if row["recall"] else "FAIL"
        elif row["score"] is None:
            score_str = "N/A"
        else:
            score_str = str(row["score"])
        print(f"{q_short:<45} {recall:>6} {score_str:>5}")

    total_answerable = sum(1 for r in results if not r["grounding_check"])
    recall_rate = recall_hits / total_answerable if total_answerable else 0.0
    mean_score = sum(ratings) / len(ratings) if ratings else 0.0
    grounding_acc = grounding_hits / grounding_total if grounding_total else 0.0

    print(f"\n{'='*58}")
    print(f"retrieval_recall@5:    {recall_rate:.2%}")
    print(f"answer_quality:        {mean_score:.2f}")
    print(f"grounding_accuracy:    {grounding_acc:.2%}")
    print(f"unparseable_responses: {unparseable_count}")


if __name__ == "__main__":
    asyncio.run(main())
