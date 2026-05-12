import json
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ..config import EMBEDDING_MODEL, KNOWLEDGE_BASE_PATH, TOP_K, MODEL
from ..llm_client import get_client

_model: SentenceTransformer | None = None
_knowledge_base: list[dict] | None = None
_doc_embeddings: np.ndarray | None = None


_model: SentenceTransformer | None = None
_knowledge_base: list[dict] | None = None
_doc_embeddings: np.ndarray | None = None
_bm25: BM25Okapi | None = None
_doc_texts: list[str] | None = None


def _tokenize(text: str) -> list[str]:
    """简单中文分词：按非字母数字汉字切分。"""
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text)


def _ensure_loaded() -> None:
    global _model, _knowledge_base, _doc_embeddings, _bm25, _doc_texts
    if _model is not None:
        return

    _model = SentenceTransformer(EMBEDDING_MODEL)
    with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
        _knowledge_base = json.load(f)

    _doc_texts = [doc["title"] + "：" + doc["content"] for doc in _knowledge_base]
    _doc_embeddings = _model.encode(_doc_texts)
    _doc_embeddings = _doc_embeddings / np.linalg.norm(_doc_embeddings, axis=1, keepdims=True)  # ← 加这行

    # BM25 索引
    tokenized = [_tokenize(t) for t in _doc_texts]
    _bm25 = BM25Okapi(tokenized)

    print(f"文档数量：{len(_knowledge_base)}")
    print(f"向量矩阵形状：{_doc_embeddings.shape}")

def _rewrite_queries(user_input: str) -> list[str]:
    """用 LLM 把用户原始输入改写成 1-3 个检索友好的 query。"""
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": (
                "你是一个查询改写器。用户会给你一段关于诈骗、钓鱼、网络安全的问题或描述。"
                "你的任务是把它改写成1到3个简短的中文检索关键词短语，用于在反诈知识库中搜索。"
                "每个短语不超过15个字，一行一个，只输出短语，不要编号，不要解释。"
                "如果原始输入已经足够简洁，直接输出原文即可。"
            )},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    queries = [q.strip() for q in raw.split("\n") if q.strip()]
    # 保底：如果 LLM 返回空，至少用原始输入
    return queries if queries else [user_input]
def search_knowledge(query: str, top_k: int = TOP_K) -> dict:
    """搜索知识库，返回结果 + 质量提示。"""
    _ensure_loaded()

    queries = _rewrite_queries(query)

    seen_titles: set[str] = set()
    all_results: list[dict] = []
    best_vec_score: float = 0.0          # ← 新增：追踪最高原始向量分

    for q in queries:
        query_embedding = _model.encode(q)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        vec_scores = np.dot(_doc_embeddings, query_embedding)

        # 记录这轮最高原始向量分
        round_best = float(vec_scores.max())
        if round_best > best_vec_score:
            best_vec_score = round_best

        bm25_scores = _bm25.get_scores(_tokenize(q))

        vec_norm = vec_scores / (vec_scores.max() + 1e-9)
        bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
        combined = 0.5 * vec_norm + 0.5 * bm25_norm

        top_indices = np.argsort(combined)[::-1][:top_k]

        for i in top_indices:
            title = _knowledge_base[i]["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            all_results.append({
                "title": title,
                "content": _knowledge_base[i]["content"],
                "score": float(combined[i]),
            })

    all_results.sort(key=lambda x: x["score"], reverse=True)
    results = all_results[:top_k]

    # —— 质量提示：告诉 LLM 这批结果够不够用 ——
    CONFIDENCE_THRESHOLD = 0.50
    if best_vec_score < CONFIDENCE_THRESHOLD:
        quality_hint = (
            f"知识库最高语义相似度仅 {best_vec_score:.2f}，低于阈值 {CONFIDENCE_THRESHOLD}。"
            "结果可能不相关，建议调用 web_search 获取更准确的信息。"
        )
    else:
        quality_hint = f"知识库最高语义相似度 {best_vec_score:.2f}，结果可信度较高。"

    return {
        "results": results,
        "quality_hint": quality_hint,
    }
