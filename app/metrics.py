"""六项指标 + bootstrap CI + 失败归因。口径照 HANDOVER 5.4，不许改。

f1         : token 级 F1（小写、去标点、按空格切）
cit_p      : citations 命中 gold_pids 的个数 / len(citations)
cit_r      : 同上 / len(gold_pids)
success    : f1 >= 0.5 and cit_p >= 0.5
p95_latency: 成功任务 latency_ms 的第 95 百分位
cost_per_task: 总 cost_cny / 成功任务数
"""
from __future__ import annotations

import random
import re

_PUNCT = re.compile(r"[^\w\s]")
_SPACE = re.compile(r"\s+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    s = (s or "").lower()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return _SPACE.sub(" ", s).strip()


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = set(p) & set(g)
    if not common:
        return 0.0
    hit = sum(min(p.count(t), g.count(t)) for t in common)
    if hit == 0:
        return 0.0
    precision, recall = hit / len(p), hit / len(g)
    return 2 * precision * recall / (precision + recall)


def citation_scores(citations, context_pids, gold_pids):
    """citations 是 1..k 的段落序号，映射回 pid 再判 gold。越界记错配但仍计入分母。"""
    gold = set(gold_pids or [])
    cites = list(citations or [])
    if not cites:
        return 0.0, 0.0
    hit = 0
    for c in cites:
        try:
            i = int(c)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(context_pids) and context_pids[i - 1] in gold:
            hit += 1
    return hit / len(cites), (hit / len(gold) if gold else 0.0)


def is_success(f1_score: float, cit_p: float) -> int:
    return 1 if (f1_score >= 0.5 and cit_p >= 0.5) else 0


def bootstrap_ci(xs, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """percentile 法，返回 (mean, lo, hi)。"""
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    m = len(xs)
    means = []
    for _ in range(n):
        means.append(sum(xs[rng.randrange(m)] for _ in range(m)) / m)
    means.sort()
    lo = means[int((alpha / 2) * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return (sum(xs) / m, lo, hi)


def paired_delta_ci(a, b, n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """配对差 a-b 的 bootstrap CI。两条 config 跑的是同一批 qid，配对比独立重采样更紧。"""
    diffs = [float(x) - float(y) for x, y in zip(a, b)]
    return bootstrap_ci(diffs, n=n, alpha=alpha, seed=seed)


def p95(xs):
    xs = sorted(float(x) for x in xs if x is not None)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))
    return xs[i]


def failure_class(rec: dict, gold_pids, f1_score: float, cit_p: float):
    """四分类：未召回 / 召回未读入 / 读入推理错 / 引用错配。正确返回 None。"""
    gold = set(gold_pids or [])
    retrieved = set(rec.get("all_retrieved_pids") or [])
    context = set(rec.get("context_pids") or [])
    if not (gold & retrieved):
        return "未召回"
    if not (gold & context):
        return "召回未读入"
    if f1_score < 0.5:
        return "读入推理错"
    if cit_p < 0.5:
        return "引用错配"
    return None
