import json

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import EMBEDDING_MODEL, KNOWLEDGE_BASE_PATH, TOP_K

_model: SentenceTransformer | None = None
_knowledge_base: list[dict] | None = None
_doc_embeddings: np.ndarray | None = None


def _ensure_loaded() -> None:
    global _model, _knowledge_base, _doc_embeddings
    if _model is not None:
        return

    _model = SentenceTransformer(EMBEDDING_MODEL)
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        _knowledge_base = json.load(f)

    doc_texts = [doc["title"] + "：" + doc["content"] for doc in _knowledge_base]
    _doc_embeddings = _model.encode(doc_texts)

    print(f"文档数量：{len(_knowledge_base)}")
    print(f"向量矩阵形状：{_doc_embeddings.shape}")


def search_knowledge(query: str, top_k: int = TOP_K) -> list[dict]:
    _ensure_loaded()
    query_embedding = _model.encode(query)
    scores = np.dot(_doc_embeddings, query_embedding)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_indices:
        results.append({
            "title": _knowledge_base[i]["title"],
            "content": _knowledge_base[i]["content"],
            "score": float(scores[i]),
        })
    return results
