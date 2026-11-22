"""
Assignment 3: evaluation (`main.py evaluate`).

Stage 1, retrieval: hit rate at k = 1, 3, 5 for each chunking design, on the
answerable questions in eval/questions.csv, plus how many words the top-k
chunks add to the prompt. Results go to eval/retrieval.csv.

A retrieved chunk is a hit when its source is one of the question's expected
sources and it touches one of the expected locations. For "multi" questions we
also report whether *all* expected passages were found.

Question vectors are cached in eval/query_vectors.json: each question costs one
embedding once, and the same vector is reused for both designs.
"""

import csv
import json
from pathlib import Path

import index

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval"
QUESTIONS = EVAL_DIR / "questions.csv"
QUERY_CACHE = EVAL_DIR / "query_vectors.json"
KS = (1, 3, 5)


def load_questions() -> list[dict]:
    with QUESTIONS.open(newline="") as f:
        return list(csv.DictReader(f))


def query_vectors(questions: list[dict]) -> dict[str, list[float]]:
    cache = json.loads(QUERY_CACHE.read_text()) if QUERY_CACHE.exists() else {}
    for q in questions:
        if q["question"] not in cache:
            cache[q["question"]] = index.embed_query(q["question"])
            QUERY_CACHE.write_text(json.dumps(cache))  # save each one: a quota stop loses nothing
    return cache


def expected_passages(q: dict) -> list[tuple[str, str]]:
    sources = q["expected_source"].split(";")
    locations = q["expected_location"].split(";")
    if len(sources) == 1:
        sources = sources * len(locations)
    return list(zip(sources, locations))


def touches(chunk: dict, passage: tuple[str, str]) -> bool:
    source, location = passage
    return chunk["source"] == source and location in chunk["locations"].split("|")


def retrieval_eval(questions: list[dict], vectors: dict) -> list[dict]:
    answerable = [q for q in questions if q["kind"] != "unanswerable"]
    rows = []
    for design in index.DESIGNS:
        hits = {k: 0 for k in KS}
        all_found = {k: 0 for k in KS}
        words = {k: 0 for k in KS}
        multi = [q for q in answerable if q["kind"] == "multi"]
        for q in answerable:
            top = index.search(q["question"], design, k=max(KS), vector=vectors[q["question"]])
            passages = expected_passages(q)
            for k in KS:
                found = [any(touches(c, p) for c in top[:k]) for p in passages]
                hits[k] += any(found)
                words[k] += sum(c["words"] for c in top[:k])
                if q["kind"] == "multi":
                    all_found[k] += all(found)
        n = len(answerable)
        for k in KS:
            rows.append({
                "design": design, "k": k,
                "hit_rate": round(hits[k] / n, 3),
                "multi_all_passages": round(all_found[k] / len(multi), 3) if multi else "",
                "avg_words_added": round(words[k] / n),
                "questions": n,
            })
    return rows


def main() -> None:
    questions = load_questions()
    vectors = query_vectors(questions)
    rows = retrieval_eval(questions, vectors)
    with (EVAL_DIR / "retrieval.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
