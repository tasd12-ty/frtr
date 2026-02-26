"""CLIP/SigLIP embedding backend using sentence-transformers."""

from __future__ import annotations

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from .base import EmbeddingBackend


class CLIPEmbedder(EmbeddingBackend):
    """Local CLIP-based multimodal embedder.

    Uses sentence-transformers which supports CLIP, SigLIP, and similar
    vision-language models. Text and images are mapped to the same latent
    space, enabling cross-modal retrieval.
    """

    def __init__(self, model_name: str = "clip-ViT-B-32") -> None:
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> np.ndarray:
        return self._model.encode(text, convert_to_numpy=True).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, convert_to_numpy=True, batch_size=64).astype(
            np.float32
        )

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self._model.encode(image, convert_to_numpy=True).astype(np.float32)
