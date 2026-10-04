"""Локальная индексация документов для RAG."""

from .service import RAG_PIPELINE_MODES, RAG_STRATEGIES, RagIndexService, build_rag_context
from .evidence import (
    EvidenceValidationError,
    RAG_UNKNOWN_RESPONSE,
    RAG_UNSUPPORTED_RESPONSE,
    apply_relevance_gate,
    build_evidence_guidance,
    parse_semantic_evaluation,
    repair_prompt,
    retry_unknown_prompt,
    semantic_evaluation_prompt,
    validate_and_format_evidence,
)

__all__ = [
    "EvidenceValidationError", "RAG_PIPELINE_MODES", "RAG_STRATEGIES",
    "RAG_UNKNOWN_RESPONSE", "RAG_UNSUPPORTED_RESPONSE", "RagIndexService",
    "apply_relevance_gate", "build_evidence_guidance", "build_rag_context",
    "parse_semantic_evaluation", "repair_prompt", "retry_unknown_prompt", "semantic_evaluation_prompt",
    "validate_and_format_evidence",
]
