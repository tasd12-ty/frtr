# FRTR: From Rows to Reasoning

A multimodal retrieval-augmented generation (RAG) framework for spreadsheet understanding.
Based on paper: [arXiv:2601.08741](https://arxiv.org/abs/2601.08741)

## Quick Start (any machine)

```bash
# 1. Clone
git clone https://github.com/tasd12-ty/frtr.git && cd frtr

# 2. Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install all dependencies (one command, uses lock file for reproducibility)
uv sync

# 4. Verify
uv run python -m frtr --help
```

`uv sync` reads `pyproject.toml` + `uv.lock`，自动创建 `.venv` 并安装所有依赖，无需手动 pip install。

## Usage

### 1. Prepare data

将 FRTR-Bench 的 `.xlsx` 文件放入某个目录:

```bash
ls /path/to/data/frtr_*.xlsx
```

### 2. Index only (offline, no LLM needed)

```bash
uv run python -m frtr --data-dir /path/to/data --index-only
```

### 3. Run benchmark with vLLM

```bash
# GPU server: start vLLM
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000

# Client: run FRTR benchmark
uv run python -m frtr \
    --data-dir /path/to/data \
    --base-url http://gpu-server:8000/v1 \
    --llm-model Qwen/Qwen2.5-VL-72B-Instruct \
    --max-questions 5
```

### 4. Full benchmark

```bash
uv run python -m frtr \
    --data-dir /path/to/data \
    --base-url http://gpu-server:8000/v1 \
    --llm-model Qwen/Qwen2.5-VL-72B-Instruct \
    --output results.json
```

### 5. Split indexing and evaluation

```bash
# Step 1: index (slow, only once)
uv run python -m frtr --data-dir /path/to/data --index-only

# Step 2: evaluate (fast, reuse index)
uv run python -m frtr \
    --data-dir /path/to/data \
    --skip-index \
    --base-url http://gpu-server:8000/v1 \
    --llm-model Qwen/Qwen2.5-VL-72B-Instruct
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir` | Directory with `.xlsx` files | auto |
| `--index-dir` | Where to store index | `.index/` |
| `--embedding` | `clip` / `openai` / `bedrock` | `clip` |
| `--llm-model` | Model name | `gpt-4o` |
| `--base-url` | vLLM / OpenAI-compatible API URL | OpenAI official |
| `--index-only` | Only build index | - |
| `--skip-index` | Use existing index | - |
| `--max-questions` | Limit questions (for testing) | all |
| `--workbook` | Single workbook filename stem | all |
| `--output` | JSON results output path | `.index/results.json` |

## Environment Variables

All config can also be set via `FRTR_` prefixed env vars:

```bash
export FRTR_LLM_BASE_URL=http://gpu-server:8000/v1
export FRTR_LLM_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
export FRTR_EMBEDDING_PROVIDER=clip
export FRTR_DATA_DIR=/path/to/data
```

## Architecture

```
Stage 1 (Offline): Excel → Rows/Columns/Windows/Images → CLIP Embedding → ChromaDB + BM25
Stage 2 (Online):  Query → Dense(top-20) + BM25(top-20) → RRF Fusion(k=60) → top-10 chunks
Stage 3 (Online):  Query + Chunks + Images → LLM → JSON {reasoning, answer}
```
