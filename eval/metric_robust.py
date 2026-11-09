"""口径稳健性：token-F1 判错但语义正确的样本有多少，是否影响 k 5→20 的 Δ。

动因（E1-5 人工抽检发现）：16 条失败样本里有 4 条答案语义正确却被判错 ——
  - gold 是一整句（"Ballarat Bitter is a 4.9% (abv) Australian beer"）模型只答实体；
  - 词形变体（gold "French" vs 答 "France"；gold "A.P. Møller" vs 答 "Arnold Peter Møller"）。
这不是随机噪声：若 k=20 更容易输出完整句，宽松口径下 Δ 会缩水。

做法：定义「宽松答案判定」= token-F1>=0.5 或（归一化后 gold 与 answer 互为子串）。
success_lenient = 宽松答案 且 cit_p>=0.5。与严格口径并排比 Δ 与 CI。

用法：python eval/metric_robust.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metrics as M  # noqa: E402
from eval import common  # noqa: E402

ART = {"a", "an", "the"}
_PUNCT = re.compile(r"[^a-z0-9\s]")


def norm(s: str) -> str:
    s = _PUNCT.sub(" ", (s or "").lower())
    return " ".join(t for t in s.split() if t not in ART)


def lenient_ok(answer: str, gold: str, f1: float) -> bool:
    if f1 >= 0.5:
        return True
    a, g = norm(answer), norm(gold)
    if not a or not g:
        return False
    # 互为子串：接住「答实体 vs gold 整句」与「答全称 vs gold 缩写」
    return (len(g) >= 3 and g in a) or (len(a) >= 3 and a in g)


def rng_for(key: str, qs: list[str]):
    h = hashlib.sha256("|".join([key, *qs]).encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(h, "big"))


def boot_delta(a: dict, b: dict, qs: list[str], key: str, n_boot: int = 2000):
    d = np.array([b[q] - a[q] for q in qs], dtype=float)
    rng = rng_for(key, qs)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    mus = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(mus, 2.5)), float(np.percentile(mus, 97.5))


def load_arm(d: str, k: int | None, seeds=(42, 43)) -> dict[str, dict]:
    per: dict[str, dict] = {}
    for sd in seeds:
        p = Path(d) / (f"hybrid_rr-on_rw-off_top{k}_seed{sd}.jsonl" if k else
                       f"hybrid_rr-on_rw-off_seed{sd}.jsonl")
        for r in common.load_records(p):
            s = SAMPLES.get(r["qid"])
            if s is None:
                continue  # 1500 题库里的 qid 不在 600 题库中，跳过
            f1 = M.f1(r["answer"], s["answer"])
            cp, _ = M.citation_scores(r["citations"], r["context_pids"], s["gold_pids"])
            slot = per.setdefault(r["qid"], {"strict": [], "lenient": [], "flip": 0})
            slot["strict"].append(M.is_success(f1, cp))
            lo = lenient_ok(r["answer"], s["answer"], f1)
            slot["lenient"].append(lo and cp >= 0.5)
            if lo and not M.is_success(f1, cp) and f1 < 0.5:
                slot["flip"] += 1
    return {q: {"strict": float(np.mean(v["strict"])), "lenient": float(np.mean(v["lenient"])),
                "flip": v["flip"]} for q, v in per.items()}


def main() -> int:
    global SAMPLES
    SAMPLES = {s["qid"]: s for s in common.load_samples(Path("data/hotpot_dev_600.jsonl"))}
    a = load_arm("eval/results_confirm_1500", None)
    b = load_arm("eval/results_topk", 20)
    qs = sorted(set(a) & set(b))
    L = ["# 口径稳健性：严格 F1 vs 宽松（互为子串）", "",
         f"n={len(qs)} 题（600 题库 ∩ 两个结果目录），seed 42/43 先按题取均值。", ""]

    L.append("## 1. 绝对水平\n")
    L.append("| 口径 | top_k=5 | top_k=20 | Δ | 95% CI | 跨 0 |")
    L.append("|---|---|---|---|---|---|")
    for key, name in (("strict", "严格（f1>=0.5 且 cit_p>=0.5）"),
                      ("lenient", "宽松（互为子串 或 f1>=0.5，且 cit_p>=0.5）")):
        aa = {q: a[q][key] for q in qs}
        bb = {q: b[q][key] for q in qs}
        mu, lo, hi = boot_delta(aa, bb, qs, f"mr:{key}")
        L.append(f"| {name} | {np.mean([aa[q] for q in qs]):.4f} | {np.mean([bb[q] for q in qs]):.4f} | "
                 f"{mu*100:+.2f} pt | [{lo*100:+.2f}, {hi*100:+.2f}] | {'**是**' if lo < 0 < hi else '否'} |")
    L.append("")

    fl5 = sum(a[q]["flip"] for q in qs)
    fl20 = sum(b[q]["flip"] for q in qs)
    L.append("## 2. 被判错但语义正确的记录数（仅统计 f1<0.5 且互为子串）\n")
    L.append(f"- top_k=5：{fl5} / {len(qs)*2} 条记录")
    L.append(f"- top_k=20：{fl20} / {len(qs)*2} 条记录")
    L.append("")
    L.append("> 该计数只覆盖「词形/整句差异」这一类，接不住 Prussian→German 这种语义近似但字面不同的情况，"
             "所以是**下界**。")
    L.append("")
    L.append("## 3. 判读\n")
    L.append("- 两个口径的 Δ 若落在彼此 CI 内 → 严格口径没有系统性放大 k 的效应，结论不依赖 F1 阈值细节。")
    L.append("- 若宽松口径 Δ 明显变小（≥2 pt）→ 说明「k=20 更爱输出完整句」带来了一部分增益，需在文档中披露。")

    out = Path("eval/results_topk/metric_robust.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[ok] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
