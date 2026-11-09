"""E1-5 失败四分类抽检：分层导出样本供人工核对（¥0，只读 jsonl）。

用法：python eval/manual_audit.py [--per-class 5] [--seed 42]
产物：eval/results_topk/manual_audit.md（含每题的问题/答案/gold/上下文全文与脚本判定）

设计原则：导出时**不写**脚本的中间结论（不给「我认为是 X 类」的理由），
只给原始事实（gold pid 列表、检索到的 pid、上下文 pid、答案、引用），
让核对者从原始事实独立判类，再与脚本类别比对 —— 否则等于用结论验证结论。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metrics as M  # noqa: E402
from eval import common  # noqa: E402

CLASSES = ("未召回", "召回未读入", "读入推理错", "引用错配")


def class_of(rec: dict, s: dict) -> str:
    """与 confirm.py 完全一致的口径（含确定性 tie-break）。"""
    gold = set(s["gold_pids"])
    got = set(rec.get("all_retrieved_pids") or [])
    ctx = set(rec.get("context_pids") or [])
    if not (gold & got):
        return "未召回"
    if not (gold & ctx):
        return "召回未读入"
    f1 = M.f1(rec["answer"], s["answer"])
    cp, _ = M.citation_scores(rec["citations"], rec["context_pids"], s["gold_pids"])
    if f1 < 0.5:
        return "读入推理错"
    if cp < 0.5:
        return "引用错配"
    return "其他"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="eval/results_confirm_1500/hybrid_rr-on_rw-off_seed42.jsonl")
    ap.add_argument("--data", default="data/hotpot_dev_600.jsonl")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--out", default="eval/results_topk/manual_audit.md")
    args = ap.parse_args()

    samples = {s["qid"]: s for s in common.load_samples(Path(args.data))}
    buckets: dict[str, list] = {c: [] for c in CLASSES}
    for rec in common.load_records(Path(args.arm)):
        s = samples.get(rec["qid"])
        if s is None:
            continue
        f1 = M.f1(rec["answer"], s["answer"])
        cp, cr = M.citation_scores(rec["citations"], rec["context_pids"], s["gold_pids"])
        if M.is_success(f1, cp):
            continue  # 只看失败样本
        c = class_of(rec, s)
        if c in buckets:
            buckets[c].append((rec, s, f1, cp, cr))

    L = ["# E1-5 失败四分类抽检样本", "",
         f"来源 `{args.arm}`，只含失败样本（success 判定不通过）。",
         "每类取前 %d 条（按文件顺序，非刻意挑选）。" % args.per_class, "",
         "**核对方法**：先只看「原始事实」区块自行判类，再看末尾的脚本判定。", ""]

    for c in CLASSES:
        items = buckets[c]
        L.append(f"## {c}（该类失败共 {len(items)} 条，抽 {min(args.per_class, len(items))} 条）\n")
        if not items:
            L.append("> 该类在本档没有样本。\n")
            continue
        for i, (rec, s, f1, cp, cr) in enumerate(items[: args.per_class], 1):
            gold = set(s["gold_pids"])
            got = set(rec.get("all_retrieved_pids") or [])
            ctx = list(rec.get("context_pids") or [])
            L.append(f"### {c}-{i}　qid `{rec['qid']}`\n")
            L.append("原始事实：\n")
            L.append(f"- 问题：{s['question']}")
            L.append(f"- 标准答案：`{s['answer']}`")
            L.append(f"- 模型答案：`{rec['answer']}`")
            L.append(f"- 答案 token-F1 = {f1:.3f}")
            L.append(f"- gold 段落 {len(gold)} 条：{sorted(gold)}")
            L.append(f"- 检索到（并集，共 {len(got)} 条）：{sorted(got)}")
            L.append(f"- 进入上下文的 {len(ctx)} 条：{ctx}")
            L.append(f"- 上下文里命中的 gold：{sorted(gold & set(ctx))}"
                     f"　漏掉的 gold：{sorted(gold - set(ctx))}")
            L.append(f"- 模型引用 {rec['citations']} → 映射到 pid "
                     f"{[ctx[i-1] for i in rec['citations'] if isinstance(i, int) and 1 <= i <= len(ctx)]}")
            L.append(f"- 引用精确率 cit_p = {cp:.3f}，召回 cit_r = {cr:.3f}")
            L.append(f"- 终止原因 `{rec['terminated_by']}`，步数 {rec['tool_calls']}")
            L.append("\n上下文原文（★ = gold）：\n")
            for j, pid in enumerate(ctx, 1):
                p = next((x for x in s["passages"] if x["pid"] == pid), None)
                if p is None:
                    continue
                star = "★" if pid in gold else "　"
                txt = " ".join(p["text"].split())[:320]
                L.append(f"{star}[{j}] ({p['title']}) {txt}")
            L.append(f"\n<details><summary>脚本判定</summary>\n\n`{c}`\n\n</details>\n")

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[ok] {args.out}")
    for c in CLASSES:
        print(f"  {c}: {len(buckets[c])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
