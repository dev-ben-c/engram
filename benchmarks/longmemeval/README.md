# LongMemEval Benchmark

Benchmarks Engram's retrieval against [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (ICLR 2025) — 500 questions testing long-term memory across 6 categories.

## Results

| Run | R@5 | Notes |
|-----|-----|-------|
| v1 (baseline) | 0.970 (485/500) | No temporal timestamps |
| v2 (temporal created_at) | 0.970 (485/500) | Correct session timestamps |
| MemPalace (raw) | 0.966 | ChromaDB + SQLite |
| MemPalace + Haiku rerank | 1.000 | With LLM reranking |

### By Question Type (v2)

| Type | R@5 | Hits/Total |
|------|-----|------------|
| knowledge-update | 1.000 | 78/78 |
| single-session-assistant | 1.000 | 56/56 |
| multi-session | 0.985 | 131/133 |
| single-session-user | 0.957 | 67/70 |
| temporal-reasoning | 0.940 | 125/133 |
| single-session-preference | 0.933 | 28/30 |

## Running

```bash
# 1. Download dataset (~277MB)
cd benchmarks/longmemeval
curl -L -o longmemeval_s_cleaned.json \
  "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"

# 2. Ensure Ollama is running with nomic-embed-text
ollama pull nomic-embed-text

# 3. Run benchmark
source ../../venv/bin/activate
python run_benchmark.py              # full 500 questions (~20 min)
python run_benchmark.py --limit 20   # quick subset
python run_benchmark.py --k 10       # R@10 instead of R@5
```

Results are saved to `results/run_<timestamp>.json`.
