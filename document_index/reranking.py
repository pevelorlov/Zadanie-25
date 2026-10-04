"""Лениво загружаемый локальный мультиязычный cross-encoder для RAG."""

from __future__ import annotations

from threading import RLock


DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._lock = RLock()

    @property
    def model(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as error:
                    raise RuntimeError("Установите sentence-transformers для локального reranker.") from error
                self._model = CrossEncoder(self.model_name, device="cpu", max_length=512)
            return self._model

    def score(self, query: str, texts: list[str], batch_size: int = 8) -> list[float]:
        if not texts:
            return []
        values = self.model.predict(
            [(query, text) for text in texts],
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in values]
