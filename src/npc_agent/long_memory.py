"""长期向量记忆：跨会话的摘要存储与检索。"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, PROJECT_ROOT

MEMORY_DIR = PROJECT_ROOT / "data" / "long_memory"
VECTORS_PATH = MEMORY_DIR / "vectors.npz"
SUMMARIES_PATH = MEMORY_DIR / "summaries.json"

_model: SentenceTransformer | None = None
_vectors: np.ndarray | None = None   # shape: (n, 384)
_summaries: list[str] | None = None


def _ensure_loaded() -> None:
    global _model, _vectors, _summaries
    if _model is not None:
        return

    _model = SentenceTransformer(EMBEDDING_MODEL)
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)

    if VECTORS_PATH.exists() and SUMMARIES_PATH.exists():
        _vectors = np.load(VECTORS_PATH)["arr_0"]
        with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
            _summaries = json.load(f)
        print(f"[long_memory] 已加载 {len(_summaries)} 条历史记忆")
    else:
        _vectors = np.empty((0, 384), dtype=np.float32)
        _summaries = []
        print("[long_memory] 无历史记忆，从零开始")


def save_memory(summary: str) -> None:
    """把一条对话摘要存入长期记忆。"""
    _ensure_loaded()
    global _vectors, _summaries

    vec = _model.encode(summary)
    vec = vec / np.linalg.norm(vec)  # 归一化
    _vectors = np.vstack([_vectors, vec.reshape(1, -1)])
    _summaries.append(summary)

    np.savez(VECTORS_PATH, _vectors)
    with open(SUMMARIES_PATH, "w", encoding="utf-8") as f:
        json.dump(_summaries, f, ensure_ascii=False, indent=2)
    print(f"[long_memory] 已保存，当前共 {len(_summaries)} 条记忆")


def recall_memory(query: str, top_k: int = 3) -> list[str]:
    """按语义相关性检索历史摘要。"""
    _ensure_loaded()
    if len(_summaries) == 0:
        return []

    q_vec = _model.encode(query)
    q_vec = q_vec / np.linalg.norm(q_vec)
    scores = np.dot(_vectors, q_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [_summaries[i] for i in top_indices if scores[i] > 0.3]