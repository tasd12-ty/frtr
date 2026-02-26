"""Abstract base class for embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from PIL import Image


class EmbeddingBackend(ABC):
    """Unified interface for text and image embeddings.

    All backends produce vectors in a shared latent space so that
    cosine similarity is meaningful across modalities.
    """

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text string, returning a 1-D float32 numpy array."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts, returning (N, D) float32 array."""

    @abstractmethod
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed a PIL Image, returning a 1-D float32 numpy array."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
