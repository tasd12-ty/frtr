"""OpenAI embedding backend for text, with CLIP fallback for images."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .base import EmbeddingBackend


class OpenAIEmbedder(EmbeddingBackend):
    """OpenAI text embeddings + local CLIP for images.

    Text is embedded via OpenAI's text-embedding-3-small (or configurable).
    Images fall back to a local CLIP model since OpenAI doesn't offer
    image embedding endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        clip_model: str = "clip-ViT-B-32",
    ) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dim: int | None = None

        # Lazy-load CLIP for image embedding
        self._clip = None
        self._clip_model_name = clip_model

    def _get_clip(self):
        if self._clip is None:
            from sentence_transformers import SentenceTransformer

            self._clip = SentenceTransformer(self._clip_model_name)
        return self._clip

    @property
    def dimension(self) -> int:
        if self._dim is None:
            test = self.embed_text("test")
            self._dim = len(test)
        return self._dim

    def embed_text(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(input=[text], model=self._model)
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        if self._dim is None:
            self._dim = len(vec)
        return vec

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # OpenAI supports batch embedding
        batch_size = 2048
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.embeddings.create(input=batch, model=self._model)
            vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
            all_vecs.extend(vecs)
        result = np.stack(all_vecs)
        if self._dim is None:
            self._dim = result.shape[1]
        return result

    def embed_image(self, image: Image.Image) -> np.ndarray:
        # OpenAI doesn't have image embeddings; use CLIP
        clip = self._get_clip()
        return clip.encode(image, convert_to_numpy=True).astype(np.float32)
