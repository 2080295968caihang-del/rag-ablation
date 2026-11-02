"""检索器：dense / bm25 / hybrid(RRF) / cross-encoder 重排 / query rewrite。

约定（与 HANDOVER 5.1 一致）：
- 候选池永远是「同一条样本自己的 passages」，不跨样本。
- 所有模型/网络异常一律 try/except 降级，不让单条任务崩掉。
- RRF k=60，不改。
"""
from __future__ import annotations

import os
import re
import threading

import numpy as np

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
RRF_K = 60

# 模型缓存固定在项目内 models/。fastembed 默认落在 %TEMP%/fastembed_cache，
# 临时目录被清掉后 warmup 会失败并把 dense/rerank 静默降级 —— 那会污染实验，
# 所以缓存跟着代码走。需要改位置时用 FASTEMBED_CACHE_PATH 覆盖。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.getenv("FASTEMBED_CACHE_PATH") or os.path.join(_ROOT, "models")

_LOCK = threading.Lock()
_EMB = None
_CE = None
_EMB_ERR = None
_CE_ERR = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())


def _get_emb():
    """惰性加载 fastembed 文本嵌入；失败一次后不再重试（避免每条样本都等超时）。"""
    global _EMB, _EMB_ERR
    with _LOCK:
        if _EMB is None and _EMB_ERR is None:
            try:
                from fastembed import TextEmbedding

                _EMB = TextEmbedding(model_name=EMB_MODEL, cache_dir=CACHE_DIR)
            except Exception as e:  # noqa: BLE001
                _EMB_ERR = f"{type(e).__name__}: {e}"
        return _EMB


def _get_ce():
    global _CE, _CE_ERR
    with _LOCK:
        if _CE is None and _CE_ERR is None:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _CE = TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=CACHE_DIR)
            except Exception as e:  # noqa: BLE001
                _CE_ERR = f"{type(e).__name__}: {e}"
        return _CE


def warmup() -> dict:
    """预热并回报模型加载状态，供 run.py 启动时打印。"""
    emb, ce = _get_emb(), _get_ce()
    return {
        "embedding_model": EMB_MODEL,
        "embedding_ok": emb is not None,
        "embedding_error": _EMB_ERR,
        "rerank_model": RERANK_MODEL,
        "rerank_ok": ce is not None,
        "rerank_error": _CE_ERR,
        "cache_dir": CACHE_DIR,
    }


def build_index(passages: list[dict]) -> dict:
    """建 BM25 + 归一化向量矩阵。一次建好，同一条样本内复用。"""
    texts = [p["text"] for p in passages]
    toks = [_tok(t) for t in texts]

    bm25 = None
    try:
        from rank_bm25 import BM25Okapi

        if toks and any(toks):
            bm25 = BM25Okapi(toks)
    except Exception:  # noqa: BLE001
        bm25 = None

    vecs = None
    emb = _get_emb()
    if emb is not None and texts:
        try:
            arr = np.asarray(list(emb.embed(texts, batch_size=32)), dtype=np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            vecs = arr
        except Exception:  # noqa: BLE001
            vecs = None

    return {"passages": passages, "bm25": bm25, "vecs": vecs, "sizes": len(texts)}


def _pack(index: dict, idxs, scores) -> list[dict]:
    out = []
    for i, s in zip(idxs, scores):
        p = index["passages"][int(i)]
        out.append({"pid": p["pid"], "title": p["title"], "text": p["text"], "score": float(s)})
    return out


def _rank_dense(index: dict, query: str) -> list[int]:
    vecs = index["vecs"]
    emb = _get_emb()
    if vecs is None or emb is None:
        return []
    try:
        q = np.asarray(list(emb.embed([query])), dtype=np.float32)[0]
        q /= np.linalg.norm(q) + 1e-12
        return list(np.argsort(-(vecs @ q))[: len(index["passages"])])
    except Exception:  # noqa: BLE001
        return []


def _rank_bm25(index: dict, query: str) -> list[int]:
    bm25 = index["bm25"]
    if bm25 is None:
        return []
    try:
        scores = bm25.get_scores(_tok(query))
        return list(np.argsort(-np.asarray(scores)))
    except Exception:  # noqa: BLE001
        return []


def _scores(index: dict, ranks: list[int]) -> np.ndarray:
    s = np.zeros(len(index["passages"]), dtype=np.float32)
    for r, i in enumerate(ranks):
        s[i] = -float(r)
    return s


def retrieve(index: dict, query: str, k: int = 5, mode: str = "dense") -> list[dict]:
    """mode ∈ {dense, bm25, hybrid}。异常或索引缺失返回 []（不抛）。"""
    if not index or not query:
        return []
    n = index["sizes"]
    k = max(0, min(k, n))
    if k == 0:
        return []

    if mode == "dense":
        vecs, emb = index["vecs"], _get_emb()
        if vecs is None or emb is None:
            return []
        try:
            q = np.asarray(list(emb.embed([query])), dtype=np.float32)[0]
            q /= np.linalg.norm(q) + 1e-12
            sim = vecs @ q
        except Exception:  # noqa: BLE001
            return []
        top = np.argsort(-sim)[:k]
        return _pack(index, top, sim[top])

    if mode == "bm25":
        ranks = _rank_bm25(index, query)
        if not ranks:
            return []
        sc = index["bm25"].get_scores(_tok(query))
        top = ranks[:k]
        return _pack(index, top, [sc[i] for i in top])

    if mode == "hybrid":
        rd, rb = _rank_dense(index, query), _rank_bm25(index, query)
        if not rd and not rb:
            return []
        rrf = np.zeros(n, dtype=np.float32)
        for ranks in (rd, rb):
            for r, i in enumerate(ranks):
                rrf[i] += 1.0 / (RRF_K + r + 1)
        top = np.argsort(-rrf)[:k]
        return _pack(index, top, rrf[top])

    raise ValueError(f"unknown mode: {mode}")


def rerank(query: str, cands: list[dict], k: int = 5) -> list[dict]:
    """对候选（run.py 会传 top-20）重排取 top-k；模型不可用时原序截断。"""
    if not cands:
        return []
    ce = _get_ce()
    if ce is None:
        return cands[:k]
    try:
        docs = [c["text"] for c in cands]
        scores = list(ce.rerank(query, docs))
        order = sorted(range(len(cands)), key=lambda i: -float(scores[i]))
        out = []
        for i in order[:k]:
            c = dict(cands[i])
            c["score"] = float(scores[i])
            out.append(c)
        return out
    except Exception:  # noqa: BLE001
        return cands[:k]


def rewrite(query: str) -> list[str]:
    """用 LLM 生成 3 条子查询（含原 query）。任何失败返回 [query]，绝不抛。"""
    import json as _json

    from app import llm

    prompt = (
        "Split the following multi-hop question into up to 3 short English search queries "
        "that jointly cover the entities and relations needed.\n"
        'Answer with strict JSON only: {"queries": ["...", "...", "..."]}\n\n'
        f"Question: {query}"
    )
    try:
        r = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_retries=2)
        txt = (r.get("text") or "").strip()
        if txt.startswith("```"):
            txt = txt.strip("`")
            txt = txt[txt.find("{"):] if "{" in txt else txt
        data = _json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        qs = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
        out = [query]
        for q in qs:
            if q.lower() != query.lower() and q not in out:
                out.append(q)
        return out[:3]
    except llm.LLMError:
        raise  # API 级失败必须上抛：降级成单查询会静默改写"改写组"的语义
    except Exception:  # noqa: BLE001
        return [query]
