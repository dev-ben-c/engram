#!/usr/bin/env python3
"""LongMemEval benchmark for Engram.

Loads each test question's haystack into a fresh Engram DB, runs recall,
and computes R@k metrics. Compares against MemPalace's published 96.6% R@5.

Usage:
    python run_benchmark.py [--limit N] [--start N] [--k 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Import engram from parent directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engram.store import MemoryStore, DuplicateMemoryError


DATASET_PATH = Path(__file__).parent / "longmemeval_s_cleaned.json"
TEST_DB = Path("/tmp/engram-bench.db")
RESULTS_DIR = Path(__file__).parent / "results"


def serialize_session(turns: list[dict]) -> str:
    """Convert a session (list of turn dicts) into a single text blob."""
    parts = []
    for turn in turns:
        role = turn.get("role", "?")
        content = turn.get("content", "").strip()
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def reset_db():
    """Wipe the test DB so each question starts fresh."""
    if TEST_DB.exists():
        TEST_DB.unlink()
    # Also wipe the WAL/SHM if present
    for suffix in ("-wal", "-shm"):
        p = TEST_DB.with_suffix(TEST_DB.suffix + suffix)
        if p.exists():
            p.unlink()


def load_haystack(store: MemoryStore, question: dict) -> int:
    """Insert all haystack sessions for a question as episodes. Returns count loaded."""
    session_ids = question["haystack_session_ids"]
    sessions = question["haystack_sessions"]
    dates = question.get("haystack_dates", [])

    loaded = 0
    seen_keys = set()
    for i, (sid, sess) in enumerate(zip(session_ids, sessions)):
        if sid in seen_keys:
            continue  # dedupe within haystack
        seen_keys.add(sid)
        content = serialize_session(sess)
        if not content.strip():
            continue
        # Truncate extremely long sessions to keep things sane
        if len(content) > 8000:
            content = content[:8000]
        date = dates[i] if i < len(dates) else None
        try:
            store.remember(
                content=content,
                memory_type="episode",  # episodes skip duplicate detection
                category="longmemeval",
                key=sid,
                tags=[question["question_id"]],
                source=date,
                model="benchmark",
            )
            # Set created_at to the actual session date for temporal scoring
            if date:
                # Convert "2023/05/20 (Sat) 02:21" to ISO 8601
                try:
                    from datetime import datetime as _dt
                    parsed = _dt.strptime(date.split(" (")[0] + " " + date.split(") ")[1],
                                          "%Y/%m/%d %H:%M")
                    iso_date = parsed.isoformat() + "+00:00"
                    store._conn.execute(
                        "UPDATE memories SET created_at = ? WHERE key = ? AND model = 'benchmark'",
                        (iso_date, sid),
                    )
                except (ValueError, IndexError):
                    pass
            loaded += 1
        except DuplicateMemoryError:
            pass
        except Exception as e:
            print(f"  ERROR loading {sid}: {e}", file=sys.stderr)
    store._conn.commit()
    return loaded


def run_question(store: MemoryStore, question: dict, k: int = 5) -> dict:
    """Run a single question: load haystack, query, check ground truth."""
    qid = question["question_id"]
    qtype = question["question_type"]
    query = question["question"]
    answer_session_ids = set(question["answer_session_ids"])

    t0 = time.time()
    loaded = load_haystack(store, question)
    load_time = time.time() - t0

    t1 = time.time()
    results = store.recall(query=query, limit=k, category="longmemeval")
    query_time = time.time() - t1

    # Check if any returned memory's key is in the ground truth
    returned_keys = [r.key for r in results]
    hit = any(key in answer_session_ids for key in returned_keys if key)
    rank = None
    for idx, key in enumerate(returned_keys):
        if key in answer_session_ids:
            rank = idx + 1
            break

    return {
        "question_id": qid,
        "question_type": qtype,
        "loaded": loaded,
        "load_time": round(load_time, 2),
        "query_time": round(query_time, 3),
        "hit": hit,
        "rank": rank,
        "returned_keys": returned_keys,
        "answer_keys": list(answer_session_ids),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only N questions (for testing)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start at question N")
    parser.add_argument("--k", type=int, default=5,
                        help="Top-k to evaluate (R@k)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/run_<timestamp>.json)")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"ERROR: Dataset not found at {DATASET_PATH}", file=sys.stderr)
        return 1

    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} questions")

    # Slice
    end = args.start + args.limit if args.limit else len(dataset)
    questions = dataset[args.start:end]
    print(f"Running {len(questions)} questions (start={args.start}, k={args.k})")

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"run_{int(time.time())}.json"

    results = []
    type_stats = defaultdict(lambda: {"hits": 0, "total": 0})
    total_load_time = 0
    total_query_time = 0
    t_start = time.time()

    for i, q in enumerate(questions):
        reset_db()
        store = MemoryStore(db_path=TEST_DB)
        try:
            result = run_question(store, q, k=args.k)
        finally:
            store._conn.close()

        results.append(result)
        type_stats[result["question_type"]]["total"] += 1
        if result["hit"]:
            type_stats[result["question_type"]]["hits"] += 1
        total_load_time += result["load_time"]
        total_query_time += result["query_time"]

        # Progress every 5 or at the end
        if (i + 1) % 5 == 0 or i == len(questions) - 1:
            elapsed = time.time() - t_start
            hits = sum(1 for r in results if r["hit"])
            r_at_k = hits / len(results)
            avg_load = total_load_time / len(results)
            eta = (elapsed / (i + 1)) * (len(questions) - i - 1)
            print(f"  [{i+1}/{len(questions)}] R@{args.k}={r_at_k:.3f} "
                  f"({hits}/{len(results)})  avg_load={avg_load:.1f}s  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    # Final stats
    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    r_at_k = hits / total if total else 0
    elapsed = time.time() - t_start

    summary = {
        "dataset": str(DATASET_PATH.name),
        "k": args.k,
        "total_questions": total,
        "hits": hits,
        f"R@{args.k}": round(r_at_k, 4),
        "avg_load_time": round(total_load_time / total, 2) if total else 0,
        "avg_query_time": round(total_query_time / total, 3) if total else 0,
        "total_elapsed": round(elapsed, 1),
        "by_type": {
            t: {
                **stats,
                f"R@{args.k}": round(stats["hits"] / stats["total"], 4) if stats["total"] else 0,
            }
            for t, stats in type_stats.items()
        },
    }

    output_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Total: {total} questions")
    print(f"R@{args.k}: {r_at_k:.4f} ({hits}/{total})")
    print(f"Avg load time: {summary['avg_load_time']}s")
    print(f"Avg query time: {summary['avg_query_time']}s")
    print(f"Total elapsed: {elapsed:.1f}s")
    print("\nBy question type:")
    for t, stats in sorted(summary["by_type"].items()):
        print(f"  {t:30s}  R@{args.k}={stats[f'R@{args.k}']:.3f}  ({stats['hits']}/{stats['total']})")
    print(f"\nResults saved to: {output_path}")
    print(f"\nMemPalace baseline: R@5 = 0.966 (raw), 1.000 (with reranking)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
