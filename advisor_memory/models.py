"""Local model clients: Ollama chat (JSON schema output), Ollama embeddings, and an ONNX cross-encoder."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
NUM_CTX = 8192          # every chat call uses the same context size, so Ollama never reloads the model
EMBED_MODEL = "nomic-embed-text"
RERANK_MODEL = "BAAI/bge-reranker-base"
MODEL_CACHE = Path(os.environ.get("ADVISOR_MEMORY_MODELS", Path.home() / "dev" / "models" / "fastembed"))


class ModelUnavailable(RuntimeError):
    """The local model server is down or keeps failing; completed work is kept and the run can resume."""


def thinks(model: str) -> bool:
    return model.startswith("qwen3")


class Chat:
    def __init__(self, model: str, temperature: float = 0.0, seed: int = 7, num_ctx: int = NUM_CTX) -> None:
        import ollama
        self.model = model
        self.client = ollama.Client(host=HOST, timeout=600)
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}

    def json(self, system: str, user: str, schema: dict, max_tokens: int = 512) -> tuple[dict, dict]:
        kw = {"think": False} if thinks(self.model) else {}
        last = None
        for attempt in range(4):
            try:
                r = self.client.chat(model=self.model, messages=[{"role": "system", "content": system},
                                                                 {"role": "user", "content": user}],
                                     format=schema, options=self.options | {"num_predict": max_tokens}, **kw)
                text = r.message.content or ""
                usage = {"prompt_tokens": r.prompt_eval_count or 0, "output_tokens": r.eval_count or 0,
                         "seconds": round((r.total_duration or 0) / 1e9, 2)}
                try:
                    return json.loads(text), usage
                except json.JSONDecodeError:
                    return {"_raw": text}, usage   # a truncated object; the caller treats it as no answer
            except Exception as exc:  # noqa: BLE001 - connection resets, 5xx, timeouts
                last = exc
                time.sleep(5 * 2 ** attempt)
        raise ModelUnavailable(f"{self.model}: {last}")


class Embedder:
    """nomic-embed-text through Ollama, with the task prefixes the model expects."""

    def __init__(self, model: str = EMBED_MODEL) -> None:
        import ollama
        self.model = model
        self.client = ollama.Client(host=HOST, timeout=600)

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):
            for attempt in range(4):
                try:
                    out += self.client.embed(model=self.model, input=texts[i:i + 32]).embeddings
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt == 3:
                        raise ModelUnavailable(f"{self.model}: {exc}") from exc
                    time.sleep(5 * 2 ** attempt)
        v = np.asarray(out, dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(["search_document: " + t for t in texts])

    def query(self, text: str) -> np.ndarray:
        return self._embed(["search_query: " + text])[0]


class Reranker:
    """BAAI/bge-reranker-base cross-encoder on CPU (ONNX, through fastembed)."""

    def __init__(self, model: str = RERANK_MODEL) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.enc = TextCrossEncoder(model_name=model, cache_dir=str(MODEL_CACHE))

    def scores(self, query: str, docs: list[str]) -> list[float]:
        return [float(s) for s in self.enc.rerank(query, docs, batch_size=16)]
