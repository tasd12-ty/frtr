"""FRTR framework configuration."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class EmbeddingProvider(str, Enum):
    CLIP = "clip"
    OPENAI = "openai"
    BEDROCK = "bedrock"


class FRTRConfig(BaseSettings):
    """Configuration for the FRTR framework.

    All settings can be overridden via environment variables prefixed with FRTR_.
    For example: FRTR_EMBEDDING_PROVIDER=openai
    """

    model_config = {"env_prefix": "FRTR_"}

    # --- Paths ---
    data_dir: Path = Field(
        default=Path(__file__).resolve().parent.parent.parent.parent,
        description="Directory containing .xlsx workbook files",
    )
    index_dir: Path = Field(
        default=Path(__file__).resolve().parent.parent.parent / ".index",
        description="Directory for persisted ChromaDB and BM25 index",
    )

    # --- Embedding ---
    embedding_provider: EmbeddingProvider = Field(
        default=EmbeddingProvider.CLIP,
        description="Which embedding backend to use",
    )
    clip_model_name: str = Field(
        default="clip-ViT-B-32",
        description="sentence-transformers model name for CLIP embeddings",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name",
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (falls back to OPENAI_API_KEY env var)",
    )
    bedrock_model_id: str = Field(
        default="amazon.titan-embed-image-v1",
        description="AWS Bedrock embedding model ID",
    )
    bedrock_region: str = Field(default="us-east-1")
    embedding_dimension: int = Field(
        default=512,
        description="Embedding vector dimension (depends on model)",
    )

    # --- Retrieval hyperparameters (Algorithm 1) ---
    k_v: int = Field(default=20, description="Top-K for dense retrieval")
    k_s: int = Field(default=20, description="Top-K for lexical (BM25) retrieval")
    k_final: int = Field(default=10, description="Top-K after RRF fusion")
    rrf_k: int = Field(default=60, description="RRF smoothing constant")
    k_target: int = Field(
        default=500,
        description="Target number of chunks per sheet (controls sliding window size)",
    )

    # --- LLM ---
    llm_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for OpenAI-compatible API (e.g. vLLM: http://host:8000/v1)",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Model name (vLLM uses the model path, e.g. Qwen/Qwen2.5-VL-72B-Instruct)",
    )
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=2048)

    # --- E2B Sandbox ---
    e2b_api_key: Optional[str] = Field(
        default=None,
        description="E2B API key (falls back to E2B_API_KEY env var)",
    )

    def get_openai_api_key(self) -> str:
        return self.openai_api_key or os.environ.get("OPENAI_API_KEY", "")

    def get_e2b_api_key(self) -> str:
        return self.e2b_api_key or os.environ.get("E2B_API_KEY", "")
