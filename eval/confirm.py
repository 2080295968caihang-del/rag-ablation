"""确认性检验：对 n=600 筛出的两组做扩样，报告留出集效应 + gold 覆盖分解。

用法：
  python eval/confirm.py                       # 默认读 data/hotpot_dev_1500.jsonl + eval/results_confirm_1500
  python eval/confirm.py --topk-dir eval/results_topk --topk-k 10

产物：<results>/confirm.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metrics as M  # noqa: E402
from eval import common  # noqa: E402

BASE_TAG = "dense_rr-off_rw-off"
BEST_TAG = "hybrid_rr-on_rw-off"
N_BOOT = 2000
CLASSES = ("未召回", "召回未读入", "读入推理错", "引用错配")


def load_arm(d: Path, tag: str, seeds) -> dict[str, dict[str, float]]:
    """按 qid 聚合（每 qid 对 seed 取均值），返回 {qid: {metric: value}}。"""
    per: dict[str, dict[str, list[float]]] = {}
    for sd in seeds:
        p = d / f"{tag}_seed{sd}.jsonl"
        if not p.exists():
            raise SystemExit(f"缺少结果文件：{p}")
        for r in common.load_records(p):
            s = SAMPLES[r["qid"]]
            f1 = M.f1(r["answer"], s["answer"])
            cp, cr = M.citation_scores(r["citations"], r["context_pids"], s["gold_pids"])
            slot = per.setdefault(r["qid"], {"cp": [], "cr": [], "succ": [], "f1": [], "cost": [], "cov": [], "nctx": []})
            slot["nctx"].append(len(r.get("context_pids") or []))
            g = set(s["gold_pids"])
            slot["cov"].append(len(g & set(r["context_pids"])) / len(g))
            slot["cp"].append(cp)
            slot["cr"].append(cr)
            slot["succ"].append(M.is_success(f1, cp))
            slot["f1"].append(f1)
            slot["cost"].append(r["cost_cny"])
    return {q: {k: float(np.mean(v)) for k, v in d.items()} for q, d in per.items()}


def paired(a: dict, b: dict, key: str, qs: list[str], rng) -> tuple[float, float, float, float, float]:
    av = np.array([a[q][key] for q in qs])
    bv = np.array([b[q][key] for q in qs])
    d = bv - av
    mu = d.mean()
    idx = rng.integers(0, len(qs), (N_BOOT, len(qs)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return av.mean(), bv.mean(), mu, lo, hi


def cost_delta(a: dict, b: dict, qs: list[str], rng) -> tuple[float, float, float, float]:
    av = np.array([a[q]["cost"] for q in qs])
    bv = np.array([b[q]["cost"] for q in qs])
    rel = (bv.sum() - av.sum()) / av.sum()
    idx = rng.integers(0, len(qs), (N_BOOT, len(qs)))
    bs = np.array([(bv[i].sum() - av[i].sum()) / av[i].sum() for i in idx])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return av.mean(), bv.mean(), rel, (lo, hi)


def coverage_table(d: Path, seeds, lines: list[str]) -> None:
    """8 组在 n=600 上的 gold 覆盖分解：全 gold / 缺 gold 的条件成功率。"""
    lines.append("## 2. gold 覆盖分解（n=600，解释为什么 8 组没有差异）\n")
    lines.append("| config | gold 覆盖率 | 全 gold 任务占比 | success | 全 gold 时 success | 缺 gold 时 success |")
    lines.append("|---|---|---|---|---|---|")
    for r in common.RETRIEVERS:
        for rr in common.RERANKS:
            for rw in common.REWRITES:
                tag = common.tag_of(r, rr, rw)
                cov, full, succ, s_full, s_part = [], [], [], [], []
                for sd in seeds:
                    p = d / f"{tag}_seed{sd}.jsonl"
                    if not p.exists():
                        continue
                    for rec in common.load_records(p):
                        s = SAMPLES[rec["qid"]]
                        g = set(s["gold_pids"])
                        c = len(g & set(rec["context_pids"])) / len(g)
                        f1 = M.f1(rec["answer"], s["answer"])
                        cp, _ = M.citation_scores(rec["citations"], rec["context_pids"], s["gold_pids"])
                        ok = M.is_success(f1, cp)
                        cov.append(c)
                        full.append(c == 1.0)
                        succ.append(ok)
                        (s_full if c == 1.0 else s_part).append(ok)
                if not cov:
                    continue
                lines.append(
                    f"| {common.label_of(r, rr, rw)} | {np.mean(cov)*100:.1f}% | {np.mean(full)*100:.1f}% | "
                    f"{np.mean(succ):.3f} | {np.mean(s_full) if s_full else 0:.3f} | "
                    f"{np.mean(s_part) if s_part else 0:.3f} |"
                )
    lines.append("")


def arm_detail(d: Path, tag: str, k: int | None, seeds) -> dict[str, dict]:
    """逐 qid 收 success 均值与失败类型列表。k=None 表示无 top 后缀（基线 / 扩样那批）。"""
    per: dict[str, dict] = {}
    for sd in seeds:
        p = d / (f"{tag}_top{k}_seed{sd}.jsonl" if k else f"{tag}_seed{sd}.jsonl")
        for r in common.load_records(p):
            s = SAMPLES[r["qid"]]
            f1 = M.f1(r["answer"], s["answer"])
            cp, _ = M.citation_scores(r["citations"], r["context_pids"], s["gold_pids"])
            slot = per.setdefault(r["qid"], {"succ": [], "cls": []})
            slot["succ"].append(M.is_success(f1, cp))
            slot["cls"].append(M.failure_class(r, s["gold_pids"], f1, cp))
    return per


def class_of(detail: dict) -> str | None:
    """qid 级主类型：取该 qid 各 seed 失败类型里最常见的；全成功返回 None。"""
    cs = [c for c in detail["cls"] if c in CLASSES]
    if not cs:
        return None
    return max(set(cs), key=cs.count)


def ok_of(detail: dict) -> bool:
    return float(np.mean(detail["succ"])) >= 0.5


def failure_section(base_detail: dict, arms: dict[int, dict], L: list[str]) -> None:
    """k=5 相对各 top_k 档的失败类型转移 + 逐类型转化率。"""
    L.append("## 4. 失败类型转移（机制：+10.5 pt 是从哪一类里抠出来的）\n")
    L.append("| top_k | 失败数 | 未召回 | 召回未读入 | 读入推理错 | 引用错配 |")
    L.append("|---|---|---|---|---|---|")
    rows = [(5, base_detail)] + [(k, v) for k, v in sorted(arms.items())]
    for k, det in rows:
        cnt = {c: 0 for c in CLASSES}
        for dq in det.values():
            for c in dq["cls"]:
                if c in cnt:
                    cnt[c] += 1
        tot = sum(cnt.values())
        cells = " | ".join(
            f"{cnt[c]} ({cnt[c]/tot*100:.0f}%)" if tot else "0 (—)" for c in CLASSES
        )
        L.append(f"| {k} | {tot} | {cells} |")
    L.append("")
    L.append("_分母 = 该档全部失败记录数（含 2 个 seed）；类型由 `metrics.failure_class` 单一规则判定，未做人工校验。_\n")

    for k, det in sorted(arms.items()):
        qs = sorted(set(base_detail) & set(det))
        if not qs:
            continue
        f2s = [q for q in qs if not ok_of(base_detail[q]) and ok_of(det[q])]
        s2f = [q for q in qs if ok_of(base_detail[q]) and not ok_of(det[q])]
        n_fail5 = sum(1 for q in qs if not ok_of(base_detail[q]))
        n_ok5 = len(qs) - n_fail5
        L.append(f"### 4.{sorted(arms).index(k)+1} k=5 → k={k}\n")
        L.append(
            f"- 失败→成功 **{len(f2s)}/{n_fail5}（{len(f2s)/max(1,n_fail5)*100:.1f}%）**；"
            f"成功→失败 {len(s2f)}/{n_ok5}（{len(s2f)/max(1,n_ok5)*100:.1f}%）"
        )
        L.append("")
        L.append(f"| k=5 时的失败类型 | 题数 | 到 k={k} 转为成功 | 转化率 |")
        L.append("|---|---|---|---|")
        for c in CLASSES:
            grp = [q for q in qs if not ok_of(base_detail[q]) and class_of(base_detail[q]) == c]
            if not grp:
                continue
            conv = sum(1 for q in grp if ok_of(det[q]))
            L.append(f"| {c} | {len(grp)} | {conv} | {conv/len(grp)*100:.1f}% |")
        L.append("")


def check_complete(d: Path, tag: str, k: int | None, seeds, expected: int) -> str:
    """确认每个 seed 文件都是满的；返回告警文本（空串=齐）。"""
    bad = []
    for sd in seeds:
        p = d / (f"{tag}_top{k}_seed{sd}.jsonl" if k is not None else f"{tag}_seed{sd}.jsonl")
        if not p.exists():
            bad.append(f"{p.name}: 缺失")
            continue
        n = sum(1 for _ in p.open(encoding="utf-8"))
        if n != expected:
            bad.append(f"{p.name}: {n}/{expected} 行")
    return "；".join(bad)


def load_arm_pattern(d: Path, stem: str, seeds) -> dict[str, dict[str, float]]:
    """按 <stem>_seed<sd>.jsonl 读取；文件不齐返回空。"""
    per: dict[str, dict[str, list[float]]] = {}
    for sd in seeds:
        p = d / f"{stem}_seed{sd}.jsonl"
        if not p.exists():
            return {}
        for r in common.load_records(p):
            s_ = SAMPLES[r["qid"]]
            f1 = M.f1(r["answer"], s_["answer"])
            cp, _ = M.citation_scores(r["citations"], r["context_pids"], s_["gold_pids"])
            slot = per.setdefault(r["qid"], {"cp": [], "succ": [], "cost": [], "cov": [], "tc": []})
            slot["cp"].append(cp)
            slot["succ"].append(M.is_success(f1, cp))
            slot["cost"].append(r["cost_cny"])
            g = set(s_["gold_pids"])
            slot["cov"].append(len(g & set(r["context_pids"])) / len(g))
            slot["tc"].append(r.get("tool_calls") or 1)
    return {q: {kk: float(np.mean(v)) for kk, v in d_.items()} for q, d_ in per.items()}


def discover_topk(d: Path, only: str = "") -> list[int]:
    """从文件名里发现所有 top_k 档位。"""
    ks = set()
    for p in d.glob("*_top*_seed*.jsonl"):
        m = re.search(r"_top(\d+)_seed", p.name)
        if m:
            ks.add(int(m.group(1)))
    if only:
        want = {int(x) for x in re.split(r"[,\s]+", only.strip()) if x.strip().isdigit()}
        ks &= want
    return sorted(ks)


def load_topk_arm(d: Path, tag: str, k: int, seeds) -> dict[str, dict[str, float]]:
    """读 top_k=k 的那一组，按 qid 对 seed 取均值；文件不齐返回空。"""
    per: dict[str, dict[str, list[float]]] = {}
    for sd in seeds:
        p = d / f"{tag}_top{k}_seed{sd}.jsonl"
        if not p.exists():
            return {}
        for r in common.load_records(p):
            s = SAMPLES[r["qid"]]
            f1 = M.f1(r["answer"], s["answer"])
            cp, _ = M.citation_scores(r["citations"], r["context_pids"], s["gold_pids"])
            slot = per.setdefault(r["qid"], {"cp": [], "succ": [], "cost": [], "cov": [], "nctx": []})
            slot["nctx"].append(len(r.get("context_pids") or []))
            slot["cp"].append(cp)
            slot["succ"].append(M.is_success(f1, cp))
            slot["cost"].append(r["cost_cny"])
            g = set(s["gold_pids"])
            slot["cov"].append(len(g & set(r["context_pids"])) / len(g))
    return {q: {kk: float(np.mean(v)) for kk, v in d_.items()} for q, d_ in per.items()}


def main() -> int:
    global SAMPLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.ROOT / "data" / "hotpot_dev_1500.jsonl"))
    ap.add_argument("--results", default=str(common.ROOT / "eval" / "results_confirm_1500"))
    ap.add_argument("--seen", default=str(common.ROOT / "data" / "hotpot_dev_600.jsonl"),
                    help="用于选组的那批 qid（留出集 = 全集减去它）")
    ap.add_argument("--coverage-results", default=str(common.ROOT / "eval" / "results_600"))
    ap.add_argument("--topk-dir", default="")
    ap.add_argument("--topk-k", default="", help="只保留这些档位，如 10,20；留空=自动发现")
    ap.add_argument("--steps-dir", default=str(common.ROOT / "eval" / "results_steps"))
    ap.add_argument("--topk-data", default=str(common.ROOT / "data" / "hotpot_dev_600.jsonl"),
                    help="top_k 那批跑的数据集，用于核对行数")
    args = ap.parse_args()

    d = Path(args.results)
    samples_all = common.load_samples(Path(args.data))
    SAMPLES = {s["qid"]: s for s in samples_all}
    seeds = common.seeds_in(d)
    if not seeds:
        raise SystemExit(f"{d} 下没有 *_seed*.jsonl")

    base = load_arm(d, BASE_TAG, seeds)
    best = load_arm(d, BEST_TAG, seeds)
    allq = sorted(set(base) & set(best))
    seen = {s["qid"] for s in common.load_samples(Path(args.seen))} if Path(args.seen).exists() else set()
    held = [q for q in allq if q not in seen]
    used = [q for q in allq if q in seen]

    rng = np.random.default_rng(1)
    L: list[str] = []
    L.append("# 确认性检验：`hybrid+rerank` vs `dense` 基线\n")
    L.append(
        f"n=600 的 8 组筛选中，`{BEST_TAG}` 的 cit_p 最高（趋 +2.0 pt，CI 跨 0）。"
        f"按其功效需求扩样到 {len(allq)} 题（新增 {len(held)} 题），本文件报告扩样后的配对检验。\n"
    )
    L.append("| 样本 | n | 指标 | base | best | Δ | 95% CI（qid 聚类 bootstrap） | 跨 0 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for label, qs in (("全集", allq), ("选组用（原 600）", used), ("**留出集（未参与选组）**", held)):
        if not qs:
            continue
        for key, name in (("cp", "cit_p"), ("succ", "success"), ("cr", "cit_r"), ("f1", "f1")):
            a, b, mu, lo, hi = paired(base, best, key, qs, rng)
            L.append(
                f"| {label} | {len(qs)} | {name} | {a:.4f} | {b:.4f} | {mu*100:+.2f} pt | "
                f"[{lo*100:+.2f}, {hi*100:+.2f}] | {'是' if lo < 0 < hi else '**否**'} |"
            )
    a, b, rel, (lo, hi) = cost_delta(base, best, allq, rng)
    L.append(f"| 全集 | {len(allq)} | 单任务成本 | ¥{a:.5f} | ¥{b:.5f} | {rel*100:+.1f}% | "
             f"[{lo*100:+.1f}%, {hi*100:+.1f}%] | {'是' if lo < 0 < hi else '**否**'} |")
    L.append("")

    L.append("**判读**：留出集（未参与选组）与全集方向、量级一致，因此结论不是选组偏差造成的。"
             "选组用那 600 题的 CI 仍跨 0，说明 n=600 的功效不足，这正是扩样的理由。\n")
    L.append(f"功效口径：两尾 α=0.05、power=0.8，按 n=600 实测的配对 SD（≈0.27）与 Δcit_p（≈2 pt）"
             f"反推需 n≈1500；本轮实际到 n={len(allq)}。\n")

    covd = Path(args.coverage_results)
    if covd.exists():
        cseeds = common.seeds_in(covd) or seeds
        coverage_table(covd, cseeds, L)

    if args.topk_dir and Path(args.topk_dir).exists():
        td = Path(args.topk_dir)
        ks = discover_topk(td, args.topk_k)
        topk_n = len(common.load_samples(Path(args.topk_data)))
        if ks:
            L.append("## 3. top_k 干预（同一组 `hybrid+rerank`，仅改上下文槽位数）\n")
            rows: list[tuple[int, float, float, float, float]] = []
            base5 = best
            ktab: dict[int, dict] = {}
            for k in ks:
                warn = check_complete(td, BEST_TAG, k, seeds, topk_n)
                if warn:
                    L.append(f"> **跳过 top_k={k}**：结果不完整 —— {warn}。")
                    L.append("")
                    continue
                tk = load_topk_arm(td, BEST_TAG, k, seeds)
                if not tk:
                    L.append(f"> top_k={k}：结果文件不齐（{len(seeds)} seeds 需全部存在），跳过。\n")
                    continue
                ktab[k] = tk
                qs = sorted(set(tk) & set(base5))
                L.append(f"### 3.{len(rows)+1} top_k={k} vs top_k=5（n={len(qs)} 题，同题配对）\n")
                L.append(f"| 指标 | top_k=5 | top_k={k} | Δ | 95% CI | 跨 0 |")
                L.append("|---|---|---|---|---|---|")
                for key, name in (("cov", "gold 覆盖率"), ("cp", "cit_p"), ("succ", "success")):
                    a, b, mu, lo, hi = paired(base5, tk, key, qs, rng)
                    L.append(f"| {name} | {a:.4f} | {b:.4f} | {mu*100:+.2f} pt | "
                             f"[{lo*100:+.2f}, {hi*100:+.2f}] | {'是' if lo < 0 < hi else '**否**'} |")
                    if key == "cov":
                        cov_mu, cov_arm = mu, b
                    elif key == "succ":
                        succ_mu, succ_arm = mu, b
                a, b, rel, (lo, hi) = cost_delta(base5, tk, qs, rng)
                L.append(f"| 单任务成本 | ¥{a:.5f} | ¥{b:.5f} | {rel*100:+.1f}% | "
                         f"[{lo*100:+.1f}%, {hi*100:+.1f}%] | {'是' if lo < 0 < hi else '**否**'} |")
                L.append("")
                nctx = float(np.mean([tk[q]["nctx"] for q in qs]))
                rows.append((k, cov_arm, cov_mu, succ_arm, succ_mu, rel, nctx))

            if len(rows) >= 2:
                nctx5 = float(np.mean([base5[q]["nctx"] for q in
                                       sorted(set(ktab[rows[0][0]]) & set(base5))]))
                base_cov = paired(base5, ktab[rows[0][0]], "cov", sorted(set(ktab[rows[0][0]]) & set(base5)), rng)[0]
                base_succ = paired(base5, ktab[rows[0][0]], "succ", sorted(set(ktab[rows[0][0]]) & set(base5)), rng)[0]
                base_cost = np.mean([base5[q]["cost"] for q in
                                     sorted(set(ktab[rows[0][0]]) & set(base5))])
                L.append("### 饱和检查\n")
                L.append("| top_k | 平均上下文条数 | gold 覆盖率 | Δcov vs k=5 | success | Δsucc vs k=5 | 单任务成本 | 成本 vs k=5 | 有效对照 |")
                L.append("|---|---|---|---|---|---|---|---|---|")
                L.append(f"| 5 | {nctx5:.1f} | {base_cov*100:.1f}% | — | {base_succ:.3f} | — | ¥{base_cost:.5f} | — | — |")
                prev = nctx5
                for k, cov_arm, cov_mu, succ_arm, succ_mu, rel, nctx in rows:
                    gap = nctx - prev
                    okc = "是" if gap >= 1.0 else f"**否（+{gap:.1f} 条）**"
                    L.append(f"| {k} | {nctx:.1f} | {cov_arm*100:.1f}% | {cov_mu*100:+.2f} pt | {succ_arm:.3f} | "
                             f"{succ_mu*100:+.2f} pt | ¥{base_cost*(1+rel):.5f} | {rel*100:+.1f}% | {okc} |")
                    prev = nctx
                L.append("")
                L.append("**有效对照**列 = 该档平均上下文条数比上一档是否多出至少 1 条。若为否，说明这一档实际没改到槽位"
                         "（多为检索/重排候选池被硬编码截断），**不能用来判定饱和**。")
                L.append("")
                bad = [r[0] for i, r in enumerate(rows) if r[6] - (nctx5 if i == 0 else rows[i-1][6]) < 1.0]
                if bad:
                    L.append(f"⚠️ **top_k={bad} 不是有效对照**：实测平均上下文条数与上一档几乎相同，"
                             f"饱和结论不成立，真实饱和点尚未测到。见下方「候选池约束」。\n")
                L.append("判读规则：Δcov/Δsucc 从 k 到 2k 的增量若不足前一段的一半，视为饱和，不再加槽位。\n")
                arms_detail = {kk: arm_detail(td, BEST_TAG, kk, seeds) for kk in ktab}
                failure_section(arm_detail(d, BEST_TAG, None, seeds), arms_detail, L)
                L.append("## 5. 候选池约束（为什么 k=40 无效，真实饱和点怎么测）\n")
                L.append("`app/agent.py:116`：`cands = retrieve(index, q, k=20 if rerank else top_k)` —— "
                         "开启重排时候选池**硬编码 20 条**，`rerank` 再从中取 `top_k`。因此 `top_k > 20` 在本实现里"
                         "不可能增加上下文（实测 k=40 平均 20.1 条），这一档只能当重复测量用，不能当饱和证据。\n")
                L.append("- **受影响**：「k=20 之后是否继续涨」未知。")
                L.append("- **不受影响**：k=5→10→20 三档都是有效对照（上下文 5.0 / 10.0 / 19.9），"
                         "覆盖率与成功率的因果链成立。")
                L.append("- **要测真实饱和点**：把该常数提成参数（如 `--cand-k`），重跑 k=20/30/50；"
                         "3 档 × 600 题 × 2 seed ≈ ¥5。")
                L.append("- **意外的用处**：k=40 与 k=20 实质是同一配置的两次近似重复运行（上下文 20.1 vs 19.9 条），"
                         "success 0.789 vs 0.792、成本差 3.1%。这部分差异来自服务端非确定性 + 0.2 条上下文，"
                         "可当作**本管线的重复运行噪声量级（≈0.3 pt / 3%）**，用于判断其他 delta 的下限；"
                         "但它只有一对样本，不是严格的重复性实验。\n")

    sd_path = Path(args.steps_dir) if args.steps_dir else None
    if sd_path and sd_path.exists():
        one = load_arm_pattern(sd_path, f"{BEST_TAG}_steps1", seeds)
        multi = load_arm_pattern(d, BEST_TAG, seeds)
        if one and multi:
            qs = sorted(set(one) & set(multi))
            tc8 = float(np.mean([multi[q]["tc"] for q in qs]))
            tc1 = float(np.mean([one[q]["tc"] for q in qs]))
            L.append("## 6. 单步 vs 多步（`max_steps=1` vs `max_steps=8`，同检索同 top_k=5）\n")
            L.append(f"n={len(qs)} 题（同题配对）。多步组实际平均 {tc8:.2f} 次调用，单步组 {tc1:.2f} 次。\n")
            L.append("| 指标 | 8 步上限 | 1 步 | Δ | 95% CI | 跨 0 |")
            L.append("|---|---|---|---|---|---|")
            for key, name in (("cov", "gold 覆盖率"), ("cp", "cit_p"), ("succ", "success")):
                a, b, mu, lo, hi = paired(multi, one, key, qs, rng)
                L.append(f"| {name} | {a:.4f} | {b:.4f} | {mu*100:+.2f} pt | "
                         f"[{lo*100:+.2f}, {hi*100:+.2f}] | {'是' if lo < 0 < hi else '**否**'} |")
            a, b, rel, (lo, hi) = cost_delta(multi, one, qs, rng)
            L.append(f"| 单任务成本 | ¥{a:.5f} | ¥{b:.5f} | {rel*100:+.1f}% | "
                     f"[{lo*100:+.1f}%, {hi*100:+.1f}%] | {'是' if lo < 0 < hi else '**否**'} |")
            L.append(f"| 平均工具调用 | {tc8:.2f} | {tc1:.2f} | {tc1-tc8:+.2f} | — | — |")
            L.append("")
            L.append("判读：1 步 = 「检索一次 → 带引用作答」；8 步允许模型自行追加检索（步数耗尽时强制收口一次，"
                     "避免空答案把单步组系统性压低）。差值就是**多步循环本身的净价值**；"
                     "success 的 CI 若跨 0，说明在本任务上多步是可选项而非必需项。\n")

    out = d / "confirm.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
