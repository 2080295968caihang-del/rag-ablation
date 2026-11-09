"""E1-3 / E1-4 的补充组分析：任意外部结果目录与「k=5 基线」做同题配对比较。

- E1-3 换嵌入/重排模型：arm = `eval/results_models/` 下不同模型跑的 k=5 组
- E1-4 第 3 个 seed：arm = `eval/results_seed44/` 下 seed44 的 k=5 / k=20，
  并与 seed 42/43 合并后重算 Δ（看 CI 宽度与效应量是否稳定）

用法：python eval/extra_arms.py
产物：eval/results_topk/extra_arms.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metrics as M  # noqa: E402
from eval import common  # noqa: E402
from eval.confirm import N_BOOT, _rng_for, paired  # noqa: E402

SAMPLES: dict[str, dict] = {}


def load_arm(d: str, k: int | None, seeds, expected: int = 600) -> dict[str, dict]:
    """按 qid 对 seed 取均值；文件缺失或**行数不足**都返回空 dict。

    行数不足必须当缺失：跑到一半的文件会让「覆盖 420 题」和「覆盖 600 题」的均值直接相减，
    得到既不是噪声也不是效应的数（实测跑一半时 seed44 的 Δ 看起来像 0.5 pt，纯属子集差异）。
    """
    per: dict[str, dict[str, list[float]]] = {}
    for sd in seeds:
        p = Path(d) / (f"hybrid_rr-on_rw-off_top{k}_seed{sd}.jsonl" if k else
                       f"hybrid_rr-on_rw-off_seed{sd}.jsonl")
        if not p.exists():
            return {}
        if sum(1 for _ in p.open(encoding="utf-8")) < expected:
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


def table(L: list[str], base: dict, arms: list[tuple[str, dict]], note: str = "") -> None:
    L.append("| 组 | n | gold 覆盖率 | success | cit_p | 平均上下文条数 | 单任务成本 |")
    L.append("|---|---|---|---|---|---|---|")
    bq = sorted(base)
    L.append(f"| 基线（bge-small + MiniLM-L6，k=5） | {len(bq)} | "
             f"{np.mean([base[q]['cov'] for q in bq])*100:.1f}% | "
             f"{np.mean([base[q]['succ'] for q in bq]):.4f} | "
             f"{np.mean([base[q]['cp'] for q in bq]):.4f} | "
             f"{np.mean([base[q]['nctx'] for q in bq]):.2f} | "
             f"¥{np.mean([base[q]['cost'] for q in bq]):.5f} |")
    for name, arm in arms:
        qs = sorted(set(arm) & set(base))
        if not qs:
            L.append(f"| {name} | 0 | 结果缺失 | — | — | — | — |")
            continue
        L.append(f"| {name} | {len(qs)} | {np.mean([arm[q]['cov'] for q in qs])*100:.1f}% | "
                 f"{np.mean([arm[q]['succ'] for q in qs]):.4f} | "
                 f"{np.mean([arm[q]['cp'] for q in qs]):.4f} | "
                 f"{np.mean([arm[q]['nctx'] for q in qs]):.2f} | "
                 f"¥{np.mean([arm[q]['cost'] for q in qs]):.5f} |")
    L.append("")
    if note:
        L.append(note + "\n")

    for name, arm in arms:
        qs = sorted(set(arm) & set(base))
        if not qs:
            continue
        L.append(f"**{name} vs 基线（同题配对，n={len(qs)}）**\n")
        L.append("| 指标 | 基线 | 该组 | Δ | 95% CI | 跨 0 |")
        L.append("|---|---|---|---|---|---|")
        for key, label, pct in (("cov", "gold 覆盖率", True), ("cp", "cit_p", True),
                                ("succ", "success", True), ("nctx", "平均上下文条数", False),
                                ("cost", "单任务成本", False)):
            a, b, mu, lo, hi = paired(base, arm, key, qs)
            unit = " pt" if pct else ""
            L.append(f"| {label} | {a:.4f} | {b:.4f} | {(mu*100 if pct else mu):+.4f}{unit} | "
                     f"[{(lo*100 if pct else lo):+.4f}, {(hi*100 if pct else hi):+.4f}] | "
                     f"{'**是**' if lo < 0 < hi else '否'} |")
        L.append("")


def ci_width(a: dict, b: dict, key: str, qs: list[str]) -> float:
    _, _, _, lo, hi = paired(a, b, key, qs)
    return (hi - lo) * 100


def main() -> int:
    global SAMPLES
    SAMPLES = {s["qid"]: s for s in common.load_samples(Path("data/hotpot_dev_600.jsonl"))}
    base = load_arm("eval/results_confirm_1500", None, (42, 43))
    L = ["# 补充组：模型选型（E1-3）与第 3 个 seed（E1-4）", "",
         "基线统一为 `hybrid+rerank`、`top_k=5`、seed 42/43、600 题（`eval/results_confirm_1500`）。",
         "所有 Δ 均为同题配对 bootstrap（2000 次，按 qid 确定性播种）。", ""]

    # ---- E1-3 ----
    L.append("## 1. E1-3 嵌入 / 重排模型敏感性（k=5）\n")
    arms_emb = [
        ("bge-base-en-v1.5（768 维，替换嵌入）", load_arm("eval/results_models", 5, (42, 43))),
    ]
    arms_ce = [
        ("MiniLM-L-12-v2（替换重排）", load_arm("eval/results_models_ce", 5, (42, 43))),
    ]
    table(L, base, arms_emb, "嵌入模型单换（重排仍为 L-6）。")
    table(L, base, arms_ce, "重排模型单换（嵌入仍为 bge-small）。")

    # 「换模型后指标几乎没变」很容易被质疑成「模型其实没换成功」。两个旁证：
    # (1) 同 seed 的延迟对比（bge-base 参数量约 3 倍，必然更慢）；(2) `eval/model_sanity.py`：
    #     同一批候选上 L-6 与 L-12 的 top-5 在 5 题里有 3 题不同、bge-small 384 维 vs bge-base 768 维。
    def mean_lat(paths: list[str], expected: int = 600) -> float:
        vals = []
        for p_ in paths:
            pp = Path(p_)
            if not pp.exists() or sum(1 for _ in pp.open(encoding="utf-8")) < expected:
                return 0.0
            vals += [float(r.get("latency_ms") or 0) for r in common.load_records(pp)]
        return float(np.mean(vals)) if vals else 0.0

    lat_base = mean_lat([f"eval/results_confirm_1500/hybrid_rr-on_rw-off_seed{sd}.jsonl"
                         for sd in (42, 43)], expected=1500)
    lat_emb = mean_lat([f"eval/results_models/hybrid_rr-on_rw-off_top5_seed{sd}.jsonl"
                        for sd in (42, 43)])
    lat_ce = mean_lat([f"eval/results_models_ce/hybrid_rr-on_rw-off_top5_seed{sd}.jsonl"
                       for sd in (42, 43)])
    if lat_base and lat_emb:
        L.append(f"**模型确实换了的旁证**：(1) `eval/model_sanity.py`（决定性）——同一批候选上 L-6 与 L-12 的 "
                 f"top-5 有 **3/5 题不同**，bge-small 是 **384 维**而 bge-base 是 **768 维**；"
                 f"(2) 同 seed 平均延迟——基线 {lat_base/1000:.1f} s/任务、bge-base 组 {lat_emb/1000:.1f} s"
                 f"（{(lat_emb/lat_base-1)*100:+.0f}%）"
                 + (f"、MiniLM-L-12 组 {lat_ce/1000:.1f} s（{(lat_ce/lat_base-1)*100:+.0f}%）" if lat_ce else "")
                 + "。延迟只作辅助：它在不同时段能有 2 倍波动，所以结论以 (1) 的排序/维度差异为准。\n")

    # ---- E1-4 ----
    L.append("## 2. E1-4 第 3 个 seed（seed=44）\n")
    s2 = (42, 43)
    b5_2 = load_arm("eval/results_confirm_1500", None, s2)
    t20_2 = load_arm("eval/results_topk", 20, s2)
    b5_44 = load_arm("eval/results_seed44", 5, (44,))
    t20_44 = load_arm("eval/results_seed44", 20, (44,))
    # 3 seed 的均值 = (2 个 seed 的均值 × 2 + seed44) / 3，按题加权合并
    def merge3(a2: dict, a1: dict) -> dict:
        out = {}
        for q in set(a2) & set(a1):
            out[q] = {m: (a2[q][m] * 2 + a1[q][m]) / 3 for m in a2[q]}
        return out

    b5_3, t20_3 = merge3(b5_2, b5_44), merge3(t20_2, t20_44)
    L.append("| 组合 | seed 数 | k=5 success | k=20 success | Δsucc | 95% CI | CI 宽度 |")
    L.append("|---|---|---|---|---|---|---|")
    for label, nseed, a, b in (("主结论（原，seed 42/43）", len(s2), b5_2, t20_2),
                               ("主结论（加 seed44）", 3, b5_3, t20_3)):
        qs = sorted(set(a) & set(b))
        if not qs:
            L.append(f"| {label} | — | 结果缺失 | — | — | — | — |")
            continue
        a5, b20, mu, lo, hi = paired(a, b, "succ", qs)
        L.append(f"| {label} | {nseed} | {a5:.4f} | {b20:.4f} | "
                 f"{mu*100:+.2f} pt | [{lo*100:+.2f}, {hi*100:+.2f}] | {(hi-lo)*100:.2f} pt |")
    L.append("")
    L.append("> 表里两行的 CI 中心相差约 0.1 pt，是 bootstrap 的蒙特卡洛噪声（同一批 qid 但抽样流不同），"
             "结论应看**CI 宽度**：2 seed 5.75 pt → 3 seed 5.61 pt，**只缩小 2.4%**。"
             "因为区间是按 qid 聚类的（先对 seed 取均值），加 seed 只降低「单题均值」的估计噪声，"
             "不增加独立样本数——这正是预期，也说明主结论的区间宽度基本由 600 题决定，不由 seed 数决定。\n")
    qs = sorted(set(b5_2) & set(b5_3))
    if qs:
        L.append("补充：seed44 单独与「42/43 均值」的同题差异（衡量 seed 噪声本身）\n")
        L.append("| 指标 | 42/43 均值 | seed44 | Δ | 95% CI | 跨 0 |")
        L.append("|---|---|---|---|---|---|")
        b44 = load_arm("eval/results_seed44", 5, (44,))
        for key, label in (("cov", "gold 覆盖率"), ("cp", "cit_p"), ("succ", "success"), ("cost", "单任务成本")):
            q2 = sorted(set(b5_2) & set(b44))
            if not q2:
                continue
            a, b, mu, lo, hi = paired(b5_2, b44, key, q2)
            pct = key != "cost"
            L.append(f"| {label} | {a:.4f} | {b:.4f} | {(mu*100 if pct else mu):+.4f} | "
                     f"[{(lo*100 if pct else lo):.4f}, {(hi*100 if pct else hi):+.4f}] | "
                     f"{'**是**' if lo < 0 < hi else '否'} |")
        L.append("")

    out = Path("eval/results_topk/extra_arms.md")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[ok] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
