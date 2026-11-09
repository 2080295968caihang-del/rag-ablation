"""E1-2 跨配置验证：top_k 5→20 的效应是否只在 hybrid+rerank 上成立。

对照设计：
  - dense/off/off：k=5 取 `eval/results_600`（阶段 1，同日同数据集），k=20 取 `eval/results_cross`
  - hybrid/on/off：同上，作为参照（与主结论同口径，但都用 results_600 基线以保持可比）

用法：python eval/cross_config.py
产物：eval/results_cross/cross_config.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metrics as M  # noqa: E402
from eval import common  # noqa: E402
from eval.confirm import paired  # noqa: E402

SAMPLES: dict[str, dict] = {}


def load(d: str, tag: str, k: int | None, seeds=(42, 43)) -> dict[str, dict]:
    per: dict[str, dict[str, list[float]]] = {}
    for sd in seeds:
        p = Path(d) / (f"{tag}_top{k}_seed{sd}.jsonl" if k else f"{tag}_seed{sd}.jsonl")
        if not p.exists():
            return {}
        for r in common.load_records(p):
            s = SAMPLES.get(r["qid"])
            if s is None:
                continue
            f1 = M.f1(r["answer"], s["answer"])
            cp, _ = M.citation_scores(r["citations"], r["context_pids"], s["gold_pids"])
            gold, ctx = set(s["gold_pids"]), set(r.get("context_pids") or [])
            slot = per.setdefault(r["qid"], {"succ": [], "cov": [], "cp": [], "nctx": [], "cost": []})
            slot["succ"].append(1.0 if M.is_success(f1, cp) else 0.0)
            slot["cov"].append(len(gold & ctx) / max(1, len(gold)))
            slot["cp"].append(cp)
            slot["nctx"].append(float(len(ctx)))
            slot["cost"].append(float(r.get("cost_cny") or 0.0))
    return {q: {m: float(np.mean(v)) for m, v in slot.items()} for q, slot in per.items()}


def main() -> int:
    global SAMPLES
    SAMPLES = {s["qid"]: s for s in common.load_samples(Path("data/hotpot_dev_600.jsonl"))}
    L = ["# E1-2 跨配置验证：top_k 5→20 是否只在 hybrid+rerank 上成立", "",
         "两个配置各用**自己的** k=5 基线（`eval/results_600`，阶段 1 同批），"
         "k=20 组为本次新跑（`eval/results_cross` / `eval/results_topk`）。",
         "同题配对 bootstrap，2000 次，按 qid 确定性播种。", ""]

    cfgs = [
        ("dense / off / off", "dense_rr-off_rw-off", "eval/results_cross"),
        ("hybrid / on / off（主结论配置）", "hybrid_rr-on_rw-off", "eval/results_topk"),
    ]
    L.append("| 配置 | n | k=5 覆盖率 | k=20 覆盖率 | Δcov | k=5 success | k=20 success | Δsucc | 95% CI | 跨 0 | 成本 Δ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for name, tag, kd in cfgs:
        a = load("eval/results_600", tag, None)
        b = load(kd, tag, 20)
        qs = sorted(set(a) & set(b))
        if not qs:
            L.append(f"| {name} | — | 结果缺失 | — | — | — | — | — | — | — | — |")
            continue
        a5c, b20c, dcov, dlo, dhi = paired(a, b, "cov", qs)
        a5s, b20s, mu, lo, hi = paired(a, b, "succ", qs)
        ca = float(np.mean([a[q]["cost"] for q in qs]))
        cb = float(np.mean([b[q]["cost"] for q in qs]))
        dcost = (cb - ca) / ca if ca else 0.0
        rows.append((name, len(qs), a5c, b20c, dcov, a5s, b20s, mu, lo, hi, dcost, qs))
        L.append(f"| {name} | {len(qs)} | {a5c*100:.1f}% | {b20c*100:.1f}% | {dcov*100:+.2f} pt | "
                 f"{a5s:.4f} | {b20s:.4f} | **{mu*100:+.2f} pt** | [{lo*100:+.2f}, {hi*100:+.2f}] | "
                 f"{'**是**' if lo < 0 < hi else '否'} | {dcost*100:+.1f}% |")
    L.append("")

    L.append("## 判读\n")
    dense = next((r for r in rows if r[0].startswith("dense")), None)
    if dense:
        mu, lo, hi = dense[7], dense[8], dense[9]
        if lo < 0 < hi:
            L.append(f"- dense 配置 Δ = {mu*100:+.2f} pt，CI [{lo*100:+.2f}, {hi*100:+.2f}] **跨 0** → "
                     "该配置下不能声称有效应，主结论必须限定在 hybrid+rerank。")
        elif mu * 100 >= 7:
            L.append(f"- dense 配置 Δ = {mu*100:+.2f} pt，CI [{lo*100:+.2f}, {hi*100:+.2f}] 不跨 0、"
                     "且量级与 hybrid 相近 → 「槽位数是主要杠杆」**跨配置成立**。")
        elif mu * 100 >= 3:
            L.append(f"- dense 配置 Δ = {mu*100:+.2f} pt，CI [{lo*100:+.2f}, {hi*100:+.2f}]：方向一致但明显更小 → "
                     "应表述为「效应在两种配置上都为正，但量级依赖检索配置」。")
        else:
            L.append(f"- dense 配置 Δ = {mu*100:+.2f} pt，明显小于 hybrid → 这是 hybrid+rerank 特有的交互，"
                     "对外必须加限定「仅该配置」，并解释机制（重排才有 20 条候选可挑）。")
    L.append("")

    out = Path("eval/results_cross/cross_config.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[ok] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
