"""success 阈值的敏感性分析（离线重算，不花钱）。

success 定义：f1 >= T 且 cit_p >= T。主表取 T=0.5。本脚本用**现有 jsonl**重算
T ∈ {0.3,0.4,0.5,0.6,0.7} 下的绝对成功率，以及 k=5 → k=20 的配对 delta 与
qid 聚类 bootstrap CI，用来回答「0.5 是不是拍脑袋选的」。

用法：
  python eval/sensitivity.py
  python eval/sensitivity.py --thresholds 0.2,0.3,0.4,0.5,0.6,0.7,0.8

产物：eval/results_topk/sensitivity.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metrics as M  # noqa: E402
from eval import common  # noqa: E402

N_BOOT = 2000
TAG = "hybrid_rr-on_rw-off"
SAMPLES: dict[str, dict] = {}


def rng_for(key: str, qs: list[str]) -> np.random.Generator:
    """与 confirm.py 同一套确定性播种：同一个量永远给同一个 CI。"""
    h = hashlib.sha256(("|".join([key, *qs])).encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(h, "big"))


def load_perqid(d: Path, k: int | None, seeds) -> dict[str, list[tuple[float, float]]]:
    per: dict[str, list[tuple[float, float]]] = {}
    for sd in seeds:
        p = d / (f"{TAG}_top{k}_seed{sd}.jsonl" if k else f"{TAG}_seed{sd}.jsonl")
        if not p.exists():
            raise SystemExit(f"缺少结果文件：{p}")
        for r in common.load_records(p):
            s = SAMPLES.get(r["qid"])
            if s is None:  # 基线文件是 1500 题，只看与 600 题的交集
                continue
            f1 = M.f1(r["answer"], s["answer"])
            cp, _ = M.citation_scores(r["citations"], r["context_pids"], s["gold_pids"])
            per.setdefault(r["qid"], []).append((f1, cp))
    return per


def succ_at(per: dict[str, list[tuple[float, float]]], t: float) -> dict[str, float]:
    return {q: float(np.mean([1.0 if (f1 >= t and cp >= t) else 0.0 for f1, cp in v]))
            for q, v in per.items()}


def paired(a: dict, b: dict, qs: list[str], key: str) -> tuple[float, float, float, float, float]:
    av = np.array([a[q] for q in qs])
    bv = np.array([b[q] for q in qs])
    d = bv - av
    rng = rng_for(key, qs)
    idx = rng.integers(0, len(qs), (N_BOOT, len(qs)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return av.mean(), bv.mean(), d.mean(), lo, hi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.ROOT / "data" / "hotpot_dev_600.jsonl"))
    ap.add_argument("--base-dir", default=str(common.ROOT / "eval" / "results_confirm_1500"))
    ap.add_argument("--topk-dir", default=str(common.ROOT / "eval" / "results_topk"))
    ap.add_argument("--k-base", type=int, default=5)
    ap.add_argument("--k-treat", type=int, default=20)
    ap.add_argument("--thresholds", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--out", default=str(common.ROOT / "eval" / "results_topk" / "sensitivity.md"))
    args = ap.parse_args()

    global SAMPLES
    SAMPLES = {s["qid"]: s for s in common.load_samples(Path(args.data))}
    seeds = common.seeds_in() or list(common.SEEDS)

    # k=5 基线取自扩样那批（无 top 后缀）；k=20 取自 results_topk
    per5 = load_perqid(Path(args.base_dir), None, seeds)
    per20 = load_perqid(Path(args.topk_dir), args.k_treat, seeds)
    qs = sorted(set(per5) & set(per20))
    if len(qs) != len(SAMPLES):
        print(f"[warn] 配对 qid {len(qs)} != 数据集 {len(SAMPLES)}", file=sys.stderr)

    ts = [float(x) for x in args.thresholds.split(",")]
    L = ["# success 阈值敏感性（离线重算，¥0）\n",
         f"success 定义：`f1 >= T 且 cit_p >= T`。数据为现有 jsonl（`{TAG}`，"
         f"{len(qs)} 题 × {len(seeds)} seed，qid 聚类配对），k={args.k_base} 基线取自扩样那批、"
         f"k={args.k_treat} 取自 `results_topk`。\n",
         "| T | k=%d success | k=%d success | Δ | 95%% CI | 跨 0 |" % (args.k_base, args.k_treat),
         "|---|---|---|---|---|---|"]
    peak = None
    for t in ts:
        a = succ_at(per5, t)
        b = succ_at(per20, t)
        # T=0.5 就是主表口径：沿用 confirm.py 的播种键，保证与 confirm.md 的数字逐位一致
        key = "paired:succ" if abs(t - 0.5) < 1e-9 else f"thr:{t:.2f}"
        m5, m20, mu, lo, hi = paired(a, b, qs, key)
        cross = "是" if lo <= 0 <= hi else "**否**"
        L.append(f"| {t:.2f} | {m5:.4f} | {m20:.4f} | {mu*100:+.2f} pt | "
                 f"[{lo*100:+.2f}, {hi*100:+.2f}] | {cross} |")
        if peak is None or mu > peak[1]:
            peak = (t, mu)  # type: ignore[assignment]
    L.append("")
    L.append(f"结论：T 从 {ts[0]:.2f} 扫到 {ts[-1]:.2f}，Δ 的最大值出现在 T={peak[0]:.2f}"
             f"（{peak[1]*100:+.2f} pt）。若各档 Δ 同号且量级接近，说明结论不依赖 0.5 这个具体取值。\n")
    L.append("_口径：Δ = 干预组 − 基线，百分数；CI 为 qid 聚类 bootstrap 2000 次，"
             "播种键含阈值（`thr:<T>`），同一 T 重算结果恒定。_")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[ok] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
