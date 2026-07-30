"""
embedder.py — OpenRouter embedder for the CocoIndex pipeline.

Implements CocoIndex's VectorSchemaProvider (so vector column dimension is
derived, never hardcoded) and Embedder protocols, batching through
`@coco.fn.as_async(batching=True)`.

Written by hand rather than using cocoindex.ops.litellm.LiteLLMEmbedder because
LiteLLM's retry keys off HTTP status codes and its own exception classes, and the
OpenRouter throttle documented below produces neither.
"""

from __future__ import annotations

import logging
import os

import cocoindex as coco
import numpy as np
from cocoindex.resources import schema as coco_schema
from numpy.typing import NDArray
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("embedder")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EMBED_TIMEOUT_SECONDS = 60.0


def _is_upstream_rate_limit(exc: BaseException) -> bool:
    """
    Detect OpenRouter's throttle-disguised-as-success response.

    OpenRouter proxies embeddings to OpenAI's real API. When the shared upstream
    pool is throttled, OpenRouter returns an HTTP 200 whose body is an
    ``{"error": {..., "code": 429}}`` object with no ``data`` array. Because the
    status is 200, no SDK-level 429 backoff ever fires; the failure instead
    surfaces as a missing ``data`` field.
    """
    return isinstance(exc, ValueError) and "no embedding data received" in str(exc).lower()


def _is_transient(exc: BaseException) -> bool:
    """
    Decide whether a failure is worth backing off and retrying.

    Covers the disguised throttle above, plus timeouts and connection faults
    (a hung request is why an explicit client timeout exists), plus real 429s
    and 5xx. Everything else — notably a 400 for an over-long input — fails fast
    rather than being retried six times pointlessly.
    """
    if _is_upstream_rate_limit(exc):
        return True

    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True

    status = getattr(exc, "status_code", None)
    return status == 429 or (isinstance(status, int) and 500 <= status < 600)


def _is_batch_composition_error(exc: BaseException) -> bool:
    """
    Decide whether a failure is caused by *what* is in the batch rather than by
    who we are, so that splitting the batch could help.

    Only input-shaped rejections qualify: 400 (over the token limit), 413 (payload
    too large), 422 (unprocessable). Credential, permission and unknown-model
    failures affect every input equally, so splitting only multiplies the wasted
    calls; 429 and 5xx are transient and already handled by the retry on `_call`.
    """
    if not isinstance(exc, APIStatusError):
        return False

    return getattr(exc, "status_code", None) in (400, 413, 422)


class OpenRouterEmbedder(coco_schema.VectorSchemaProvider):
    """Batched OpenRouter embedder exposing its vector dimension to CocoIndex."""

    def __init__(self, model: str, api_key: str, base_url: str = OPENROUTER_BASE_URL) -> None:
        self._model = model
        # An explicit timeout is load-bearing: the SDK default is 600s, and a
        # request that never returns stalls the whole ingest silently at 0% CPU
        # with no error to retry on. 60s is generous for a 64-item batch.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=EMBED_TIMEOUT_SECONDS,
            max_retries=0,  # retries are owned by tenacity/_call below
        )
        self._dim: int | None = None

    @retry(
        retry=retry_if_exception(_is_transient),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch, retrying the upstream throttle with exponential backoff."""
        response = await self._client.embeddings.create(model=self._model, input=texts)

        # OpenRouter's throttled 200 has no usable `data`; normalise it into the
        # ValueError signature `_is_upstream_rate_limit` recognises so the retry
        # above engages instead of the caller seeing an opaque parse failure.
        data = getattr(response, "data", None)
        if not data:
            raise ValueError("No embedding data received")

        return [item.embedding for item in data]

    @coco.fn(memo=True)
    async def __coco_vector_schema__(self) -> coco_schema.VectorSchema:
        """
        Report vector dimension to CocoIndex, probing once and caching.

        The API exposes no dimension endpoint, so the dimension is measured by
        embedding a throwaway string.
        """
        if self._dim is None:
            self._dim = len((await self._call(["dimension probe"]))[0])
        return coco_schema.VectorSchema(dtype=np.dtype(np.float32), size=self._dim)

    def __coco_memo_key__(self) -> object:
        """Key memoized embeddings on the model, so switching models re-embeds."""
        return (self._model,)

    @coco.fn.as_async(batching=True, max_batch_size=64)
    async def _embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Per-batch body; concurrent `embed` calls are grouped into this."""
        try:
            vectors = await self._call(texts)
        except Exception as exc:
            # One rejected input (e.g. over the model's token limit) would
            # otherwise fail every other chunk batched alongside it. Ask the
            # engine to halve the batch and retry so the damage is isolated to
            # the offending chunk; at size 1 the engine re-raises the original.
            if _is_batch_composition_error(exc):
                raise coco.RetryWithSmallerBatch() from exc
            raise

        return [np.array(vec, dtype=np.float32) for vec in vectors]

    @coco.fn(memo=True, version=1, logic_tracking="self")
    async def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a single text. Memoized, so unchanged chunks are never re-sent."""
        return await self._embed_batch(text)

    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query, bypassing the batching and memoization machinery.

        The ingest path wants both; a one-off search wants neither — batching has
        nothing to group with, and caching a user's ad-hoc question is pointless.
        This also keeps the query path free of any CocoIndex runtime dependency.
        """
        return (await self._call([text]))[0]


def build_embedder() -> OpenRouterEmbedder:
    """Construct the embedder from environment configuration."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    return OpenRouterEmbedder(
        model=os.getenv("EMBED_MODEL", "openai/text-embedding-3-small"),
        api_key=api_key,
    )
