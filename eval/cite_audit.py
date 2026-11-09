"""引用错配的可疑率：被引段落真的不支持答案吗？

背景（E1-5 人工抽检）：`引用错配` 这类的判定是「cit_p < 0.5」，而 cit_p 的分母只认**标注的 gold 段落**。
抽检里出现过两条「引用了确实支持答案、但不在 gold 标注集里的段落」被判为引用错配 ——
HotpotQA 的 gold 是人工标注的 2 个支撑段，不等于唯一能支撑答案的段落。

本脚本给出一个**上界估计**：在 `引用错配` 的记录里，有多少条的被引（非 gold）段落文本中
**字面包含标准答案**。这部分至少可以怀疑是标注口径问题，而不是模型乱引。

用法：python eval/cite_audit.py
产物：eval/results_topk/cite_audit.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metrics as M  # noqa: E402
from eval import common  # noqa: E402

_PUNCT = re.compile(r"[^a-z0-9\s]")


def norm(s: str) -> str:
    s = _PUNCT.sub(" ", (s or "").lower())
    return " ".join(t for t in s.split() if t not in {"a", "an", "the"})


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - r) / d, (c + r) / d


def main() -> int:
    # 用 1,500 题库做全集，才能和 confirm.md §4 的记录数对上（600 题库只是它的子集）
    samples = {s["qid"]: s for s in common.load_samples(Path("data/hotpot_dev_1500.jsonl"))}
    arms = [
        ("top_k=5", "eval/results_confirm_1500", None),
        ("top_k=20", "eval/results_topk", 20),
    ]
    L = ["# 引用错配的可疑率（被引段落是否字面包含标准答案）", "",
         "只统计被 `metrics.failure_class` 判为 `引用错配` 的记录（f1>=0.5 且 cit_p<0.5）。",
         "「包含答案」= 归一化后，被引段落的非 gold 段落文本中出现标准答案字符串（长度>=3）。", ""]
    L.append("| 档位 | 引用错配记录 | 被引段落含标准答案 | 占比（Wilson 95%） |")
    L.append("|---|---|---|---|")
    for name, d, k in arms:
        # 必须是主结论那一组：results_confirm_1500 里同时有 dense 与 hybrid 两套结果，
        # 用 *_seed*.jsonl 会两套都算进来（实测把记录数从 55 虚增到 111）。
        pat = f"hybrid_rr-on_rw-off_top{k}_seed*.jsonl" if k else "hybrid_rr-on_rw-off_seed*.jsonl"
        files = sorted(Path(d).glob(pat))
        files = [p for p in files if "goldinj" not in p.name]
        n = hit = 0
        for p in files:
            for r in common.load_records(p):
                s = samples.get(r["qid"])
                if s is None:
                    continue
                f1 = M.f1(r["answer"], s["answer"])
                cp, _ = M.citation_scores(r["citations"], r["context_pids"], s["gold_pids"])
                # 必须用同一套分类器：f1>=0.5 且 cit_p<0.5 的记录若 gold 没进上下文，
                # 会先被判成「召回未读入」，直接按阈值筛会多算（实测 29 vs 27）。
                if M.failure_class(r, s["gold_pids"], f1, cp) != "引用错配":
                    continue
                n += 1
                gold = set(s["gold_pids"])
                ctx = list(r.get("context_pids") or [])
                g = norm(s["answer"])
                if len(g) < 3:
                    continue
                texts = {p_["pid"]: norm(p_["text"]) for p_ in s["passages"]}
                cited = [ctx[i - 1] for i in r["citations"]
                         if isinstance(i, int) and 1 <= i <= len(ctx)]
                if any(pid not in gold and g in texts.get(pid, "") for pid in cited):
                    hit += 1
        lo, hi = wilson(hit, n)
        L.append(f"| {name} | {n} | {hit} | {hit/max(1,n)*100:.1f}% [{lo*100:.1f}%, {hi*100:.1f}%] |")
    L.append("")
    L.append("> 这是**上界**：段落里出现答案字符串也可能只是巧合（答案常是常见实体/年份）。"
             "它只能说明「至少有一部分被判引用错配的记录，其引用并非无据」，不能直接当作误判率。")
    L.append("> 反过来的漏判（引用了 gold 段落但答案与之无关）不在本统计范围内。")

    out = Path("eval/results_topk/cite_audit.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[ok] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
