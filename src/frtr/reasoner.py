"""Stage 3: LLM reasoning and answer composition.

Implements Algorithm 1, lines 21-24:
  Construct prompt P ← {q, C, instruction}
  response ← M(P)
  return Parse JSON: {reasoning, answer}
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Optional

from .config import FRTRConfig
from .prompt_template import build_prompt
from .retriever import RetrievedChunk


@dataclass
class ReasoningResult:
    """Output from the LLM reasoning stage."""

    question: str
    answer: str
    reasoning: str
    raw_response: str
    chunks_used: list[RetrievedChunk]
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int


class LLMReasoner:
    """LLM-based answer composer using OpenAI API.

    Takes retrieved chunks and a query, constructs the prompt from
    Appendix C, and calls the LLM to produce a structured JSON answer.
    """

    def __init__(self, config: FRTRConfig) -> None:
        from openai import OpenAI

        client_kwargs = {"api_key": config.get_openai_api_key() or "EMPTY"}
        if config.llm_base_url:
            client_kwargs["base_url"] = config.llm_base_url

        self._client = OpenAI(**client_kwargs)
        self._config = config

    def reason(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        image_data: Optional[dict[str, bytes]] = None,
    ) -> ReasoningResult:
        """Run the LLM reasoning stage.

        Args:
            query:      The natural language question
            chunks:     Retrieved chunks from Stage 2
            image_data: Optional mapping of image_id -> PNG bytes for
                        image chunks (passed as vision attachments)

        Returns:
            ReasoningResult with answer, reasoning, and metrics
        """
        prompt_text = build_prompt(query, chunks)

        # Build messages
        messages = []
        content_parts = [{"type": "text", "text": prompt_text}]

        # Attach images for image-type chunks (Section 3.5)
        if image_data:
            for chunk in chunks:
                if chunk.unit_type == "image":
                    img_id = chunk.metadata.get("image_id", "")
                    if img_id in image_data:
                        b64 = base64.b64encode(image_data[img_id]).decode("utf-8")
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64}",
                                    "detail": "high",
                                },
                            }
                        )

        messages.append({"role": "user", "content": content_parts})

        # Call LLM
        start = time.time()
        response = self._client.chat.completions.create(
            model=self._config.llm_model,
            messages=messages,
            temperature=self._config.llm_temperature,
            max_tokens=self._config.llm_max_tokens,
        )
        latency = time.time() - start

        raw = response.choices[0].message.content.strip()

        # Parse JSON response
        answer = ""
        reasoning = ""
        try:
            # Handle potential markdown wrapping
            cleaned = raw
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            answer = parsed.get("answer", "")
            reasoning = parsed.get("reasoning", "")
        except json.JSONDecodeError:
            # If JSON parsing fails, use raw response
            answer = raw
            reasoning = "Failed to parse structured JSON response"

        usage = response.usage
        return ReasoningResult(
            question=query,
            answer=str(answer),
            reasoning=str(reasoning),
            raw_response=raw,
            chunks_used=chunks,
            latency_seconds=latency,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
