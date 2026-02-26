"""AWS Bedrock Titan Multimodal embedding backend (paper original)."""

from __future__ import annotations

import base64
import io
import json

import numpy as np
from PIL import Image

from .base import EmbeddingBackend


class BedrockEmbedder(EmbeddingBackend):
    """Amazon Titan Multimodal Embeddings via AWS Bedrock.

    This is the embedding backend used in the original FRTR paper.
    Requires valid AWS credentials configured via environment or
    ~/.aws/credentials.
    """

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-image-v1",
        region: str = "us-east-1",
    ) -> None:
        import boto3

        self._client = boto3.client(
            "bedrock-runtime", region_name=region
        )
        self._model_id = model_id
        self._dim = 1024  # Titan Multimodal default

    @property
    def dimension(self) -> int:
        return self._dim

    def _invoke(self, body: dict) -> np.ndarray:
        response = self._client.invoke_model(
            modelId=self._model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())
        return np.array(result["embedding"], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return self._invoke({"inputText": text})

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # Bedrock doesn't support batch; iterate
        return np.stack([self.embed_text(t) for t in texts])

    def embed_image(self, image: Image.Image) -> np.ndarray:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return self._invoke({"inputImage": b64})
