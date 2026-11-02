"""产出 eval/results/summary.md：主表 + delta 表（带 95% CI）+ 失败四分类 + gold 注入对照。

统计口径说明（写进 summary，不藏）：
  temperature=0 且检索确定性，seed 不改变轨迹 —— 3 个 seed 的逐条答案实际是同一份。
  因此 CI 按 qid 聚类（n=100，每个 qid 先对 seed 取均值）做 bootstrap，
  不把 3 份重复当成 300 个独立样本。这样 CI 只会偏保守，不会偏好看。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import common  # noqa: E402
from app import llm as llm_mod  # noqa: E402
from app import metrics  # noqa: E402

N_BOOT = 1000
ALPHA = 0.05
KEYS = ("success", "cit_p", "cit_r", "f1", "p95_ms", "cost_cny")


def per_task(rec: dict, sample: dict) -> dict:
    f1 = metrics.f1(rec.get("answer", ""), sample["answer"])
    cit_p, cit_r = metrics.citation_scores(
        rec.get("citations"), rec.get("context_pids"), sample["gold_pids"]
    )
    return {
        "qid": rec["qid"],
        "success": metrics.is_success(f1, cit_p),
        "cit_p": cit_p,
        "cit_r": cit_r,
        "f1": f1,
        "latency_ms": rec.get("latency_ms", 0),
        "cost": rec.get("cost_cny", 0.0),
        "failure": metrics.failure_class(rec, sample["gold_pids"], f1, cit_p),
    }


def collect(tag: str, samples: dict, seeds) -> list[dict]:
    rows = []
    for seed in seeds:
        for rec in common.load_records(common.result_path(tag, seed)):
            s = samples.get(rec["qid"])
            if s is None:
                continue
            r = per_task(rec, s)
            r["seed"] = seed
            rows.append(r)
    return rows


def cluster_by_qid(rows: list[dict]) -> list[dict]:
    """把同 qid 的多个 seed 合成一行（指标取均值，latency/cost 取均值）。"""
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["qid"], []).append(r)
    out = []
    for qid in sorted(by):
        g = by[qid]
        n = len(g)
        out.append(
            {
                "qid": qid,
                "success": sum(x["success"] for x in g) / n,
                "cit_p": sum(x["cit_p"] for x in g) / n,
                "cit_r": sum(x["cit_r"] for x in g) / n,
                "f1": sum(x["f1"] for x in g) / n,
                "latency_ms": sum(x["latency_ms"] for x in g) / n,
                "cost": sum(x["cost"] for x in g) / n,
                "failures": [x["failure"] for x in g],
            }
        )
    return out


def agg(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {k: 0.0 for k in KEYS}
    succ_lat = [r["latency_ms"] for r in rows if r["success"] >= 0.5]
    n_succ = sum(r["success"] for r in rows)
    return {
        "success": sum(r["success"] for r in rows) / n,
        "cit_p": sum(r["cit_p"] for r in rows) / n,
        "cit_r": sum(r["cit_r"] for r in rows) / n,
        "f1": sum(r["f1"] for r in rows) / n,
        "p95_ms": metrics.p95(succ_lat),
        "cost_cny": (sum(r["cost"] for r in rows) / n_succ) if n_succ else 0.0,
        "n": n,
        "n_succ": n_succ,
        "total_cost": sum(r["cost"] for r in rows),
    }


def boot_deltas(a: list[dict], b: list[dict], n: int = N_BOOT, seed: int = 0) -> dict:
    """配对 bootstrap：a - b，返回每个指标的 (delta, lo, hi)。"""
    pairs = {r["qid"]: r for r in a}
    pa, pb = [], []
    for r in b:
        if r["qid"] in pairs:
            pb.append(r)
            pa.append(pairs[r["qid"]])
    m = len(pa)
    if not m:
        return {k: (0.0, 0.0, 0.0) for k in KEYS}
    rng = random.Random(seed)
    acc = {k: [] for k in KEYS}
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        aa = agg([pa[i] for i in idx])
        bb = agg([pb[i] for i in idx])
        for k in KEYS:
            acc[k].append(aa[k] - bb[k])
    fa, fb = agg(pa), agg(pb)
    out = {}
    for k in KEYS:
        xs = sorted(acc[k])
        lo = xs[int((ALPHA / 2) * n)]
        hi = xs[min(n - 1, int((1 - ALPHA / 2) * n))]
        out[k] = (fa[k] - fb[k], lo, hi)
    return out


def fmt_pct(x: float) -> str:
    return f"{x:.3f}"


def fmt_pt(x: float) -> str:
    return f"{x*100:+.1f} pt"


def fmt_ci(x: tuple, scale: float = 1.0, unit: str = "") -> str:
    d, lo, hi = x
    return f"{d*scale:+.1f}{unit} [{lo*scale:+.1f}, {hi*scale:+.1f}]"


def sig(x: tuple) -> bool:
    return not (x[1] <= 0.0 <= x[2])


CLASSES = ("未召回", "召回未读入", "读入推理错", "引用错配")


def failure_counts(clust: list[dict]) -> tuple[dict, int]:
    cnt = {c: 0 for c in CLASSES}
    for r in clust:
        for f in r["failures"]:
            if f in cnt:
                cnt[f] += 1
    return cnt, sum(cnt.values())


def share_txt(cnt: dict, tot: int, only_top: bool = False) -> str:
    if not tot:
        return "无失败"
    if only_top:
        c, n = max(cnt.items(), key=lambda kv: kv[1])
        return f"主因 {c} {n/tot*100:.0f}%"
    return "、".join(f"{c} {cnt[c]/tot*100:.0f}%" for c in CLASSES if cnt[c])


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(common.DATA))
    ap.add_argument("--results", default="", help="结果目录，默认 eval/results")
    ap.add_argument("--out", default="", help="summary 输出路径，默认 <results>/summary.md")
    args = ap.parse_args()
    common.set_results(args.results or (common.ROOT / "eval" / "results"))

    samples = {s["qid"]: s for s in common.load_samples(Path(args.data))}
    seeds = list(common.SEEDS)
    combos = [(r, rr, rw) for r in common.RETRIEVERS for rr in common.RERANKS for rw in common.REWRITES]

    data = {}
    for combo in combos:
        tag = common.tag_of(*combo)
        rows = collect(tag, samples, seeds)
        if not rows:
            print(f"[warn] 缺结果：{tag}（跳过）", file=sys.stderr)
            continue
        data[combo] = {"tag": tag, "rows": rows, "clust": cluster_by_qid(rows), "agg": agg(cluster_by_qid(rows))}

    if not data:
        print("没有任何结果文件，先跑 eval/sweep.py", file=sys.stderr)
        return 1

    base_combo = common.BASE
    base = data.get(base_combo)
    lines: list[str] = []
    lines.append("# Agent 检索策略消融（HotpotQA distractor / 100 条）\n")
    n_rec = len(next(iter(data.values()))["rows"])
    lines.append(
        f"每组 {len(seeds)} seed × 100 条 = {n_rec} 条记录（{len(data)}/8 组有结果）。"
        f"每任务 LLM 温度 0、top_k=5、max_steps=8，模型 `{llm_mod.MODEL}`"
        f"（实测由 deepseek-flash 承接）。\n"
    )

    # ---- 主表 ----
    lines.append("## 1. 主表\n")
    lines.append("| config | success | cit_p | cit_r | f1 | p95_ms | cost_cny | Δsuccess vs base [95%CI] |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for combo in combos:
        if combo not in data:
            continue
        a = data[combo]["agg"]
        if base and combo != base_combo:
            d = boot_deltas(data[combo]["clust"], base["clust"], seed=42)[ "success"]
            cell = fmt_ci(d, 100, " pt")
        else:
            cell = "—"
        lines.append(
            f"| {common.label_of(*combo)} | {fmt_pct(a['success'])} | {fmt_pct(a['cit_p'])} | "
            f"{fmt_pct(a['cit_r'])} | {fmt_pct(a['f1'])} | {a['p95_ms']:.0f} | {a['cost_cny']:.4f} | {cell} |"
        )
    lines.append("")

    # ---- delta 表 ----
    lines.append("## 2. 相对基线 delta（配对 bootstrap，95% CI）\n")
    lines.append("| config | Δsuccess | Δcit_p | Δcit_r | Δf1 | Δp95_ms | Δcost |")
    lines.append("|---|---|---|---|---|---|---|")
    for combo in combos:
        if combo not in data or combo == base_combo or base is None:
            continue
        d = boot_deltas(data[combo]["clust"], base["clust"], seed=42)
        dc = d["cost_cny"][0] / base["agg"]["cost_cny"] * 100 if base["agg"]["cost_cny"] else 0.0
        dl, dh = (
            (d["cost_cny"][1] / base["agg"]["cost_cny"] * 100, d["cost_cny"][2] / base["agg"]["cost_cny"] * 100)
            if base["agg"]["cost_cny"] else (0.0, 0.0)
        )
        lines.append(
            f"| {common.label_of(*combo)} | {fmt_ci(d['success'], 100, ' pt')} | {fmt_ci(d['cit_p'], 100, ' pt')} | "
            f"{fmt_ci(d['cit_r'], 100, ' pt')} | {fmt_ci(d['f1'], 100, ' pt')} | "
            f"{fmt_ci(d['p95_ms'], 1, ' ms')} | {dc:+.1f}% [{dl:+.1f}, {dh:+.1f}] |"
        )
    lines.append("")

    # ---- 失败四分类 ----
    lines.append("## 3. 失败四分类占比（分母 = 该组失败任务数）\n")
    lines.append("| config | 失败数 | 未召回 | 召回未读入 | 读入推理错 | 引用错配 |")
    lines.append("|---|---|---|---|---|---|")
    classes = ["未召回", "召回未读入", "读入推理错", "引用错配"]
    for combo in combos:
        if combo not in data:
            continue
        cnt = {c: 0 for c in classes}
        for r in data[combo]["clust"]:
            for f in r["failures"]:
                if f in cnt:
                    cnt[f] += 1
        tot = sum(cnt.values())
        cells = " | ".join(f"{cnt[c]} ({cnt[c]/tot*100:.1f}%)" if tot else "0 (—)" for c in classes)
        lines.append(f"| {common.label_of(*combo)} | {tot} | {cells} |")
    lines.append("")

    # ---- gold 注入对照 ----
    lines.append("## 4. gold 注入对照（把 gold 段落直接塞进上下文）\n")
    inj_rows = []
    for seed in seeds:
        for rec in common.load_records(common.result_path(common.tag_of(*base_combo, gold=True), seed)):
            s = samples.get(rec["qid"])
            if s:
                inj_rows.append(per_task(rec, s))
    if inj_rows:
        ia = agg(cluster_by_qid(inj_rows))
        ba = base["agg"] if base else {"success": 0, "cit_p": 0}
        lines.append("| 组 | success | cit_p | Δsuccess vs base |")
        lines.append("|---|---|---|---|")
        lines.append(f"| base（检索侧） | {fmt_pct(ba['success'])} | {fmt_pct(ba['cit_p'])} | — |")
        lines.append(f"| gold 注入 | {fmt_pct(ia['success'])} | {fmt_pct(ia['cit_p'])} | "
                     f"{(ia['success']-ba['success'])*100:+.1f} pt |")
        gap = ia["success"] - ba["success"]
        lines.append(
            f"\n读入上限与实跑差距 **{gap*100:+.1f} pt**：这部分是检索/重排造成的损失；"
            f"gold 注入后仍失败的 {(1-ia['success'])*100:.1f}% 属于读入后的推理与引用侧问题。\n"
        )
    else:
        lines.append("_未跑 gold 注入对照（`python eval/sweep.py --gold --only dense --seeds 42`）。_\n")

    # ---- 结论 ----
    lines.append("## 5. 结论\n")
    ranked = sorted(data.items(), key=lambda kv: -kv[1]["agg"]["cit_p"])
    best_combo, best = ranked[0]
    lines.append(
        f"- cit_p 最高组：**{common.label_of(*best_combo)}**，cit_p={best['agg']['cit_p']:.3f}，"
        f"success={best['agg']['success']:.3f}，cost={best['agg']['cost_cny']:.4f} 元/成功任务。"
    )
    if base and best_combo != base_combo:
        d = boot_deltas(best["clust"], base["clust"], seed=42)
        for key, name in (("success", "成功率"), ("cit_p", "引用精确率"), ("cit_r", "引用召回率")):
            dd = d[key]
            if sig(dd):
                lines.append(f"- {name}相对基线 {dd[0]*100:+.1f} pt，95% CI [{dd[1]*100:+.1f}, {dd[2]*100:+.1f}]，**CI 不跨 0**。")
            else:
                lines.append(
                    f"- {name}相对基线 {dd[0]*100:+.1f} pt，95% CI [{dd[1]*100:+.1f}, {dd[2]*100:+.1f}]，"
                    f"**CI 跨 0，不构成结论**（趋势方向 {'正' if dd[0] > 0 else '负'}）。"
                )
        cost_chg = (best["agg"]["cost_cny"] - base["agg"]["cost_cny"]) / base["agg"]["cost_cny"] * 100 if base["agg"]["cost_cny"] else 0
        lines.append(f"- 单任务成本变化 {cost_chg:+.1f}%。")
    fb = {c: 0 for c in classes}
    for r in best["clust"]:
        for f in r["failures"]:
            if f in fb:
                fb[f] += 1
    tot = sum(fb.values())
    if tot:
        top = max(fb.items(), key=lambda kv: kv[1])
        lines.append(f"- 该组失败主因：**{top[0]}**，占失败任务 {top[1]/tot*100:.1f}%。")
    # seed 一致性实测（不要假设，跑一遍数出来）
    agree_a = agree_c = n_cmp = 0
    if base_combo in data:
        by_seed = {}
        for seed in seeds:
            by_seed[seed] = {
                r["qid"]: r for r in common.load_records(common.result_path(common.tag_of(*base_combo), seed))
            }
        qids = set.intersection(*[set(v) for v in by_seed.values()]) if by_seed else set()
        n_cmp = len(qids)
        for q in qids:
            ans = {by_seed[s][q]["answer"] for s in seeds}
            cit = {tuple(by_seed[s][q]["citations"]) for s in seeds}
            agree_a += len(ans) == 1
            agree_c += len(cit) == 1
    lines.append(
        f"\n_统计口径_：temperature=0 但服务端非严格确定性 —— base 组 3 个 seed 逐条答案相同 "
        f"{agree_a}/{n_cmp}、引用相同 {agree_c}/{n_cmp}，即 seed 带来约 {(1-agree_a/max(1,n_cmp))*100:.0f}% 的轨迹扰动。"
        f"为避免把强相关样本当独立观测，CI 按 qid 聚类（n={len(next(iter(data.values()))['clust'])}）"
        f"做 {N_BOOT} 次配对 bootstrap，区间偏保守。_\n"
    )

    # ---- 简历回填 ----
    lines.append("## 6. 简历回填（6 个字段，附来源）\n")
    mp = common.RESULTS / "manifest.json"
    total_cost = 0.0
    if mp.exists():
        import json as _json

        total_cost = float(_json.loads(mp.read_text(encoding="utf-8")).get("total_cost_cny", 0.0))
    best_rid = f"{best['tag']}@seeds{','.join(str(s) for s in seeds)}"
    base_rid = f"{base['tag']}@seeds{','.join(str(s) for s in seeds)}" if base else "—"
    lines.append("| # | 字段 | 取值 | 来源 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 1 | 实验总成本 | ¥{total_cost:.2f} | manifest.json → total_cost_cny |")
    lines.append(f"| 2 | 最优组合引用准确性 | {best['agg']['cit_p']*100:.1f}% | summary.md 主表 cit_p 最高行（{best_rid}） |")
    if base:
        lines.append(f"| 3 | 基线引用准确性 | {base['agg']['cit_p']*100:.1f}% | 主表 base 行（{base_rid}） |")
    if base and best_combo != base_combo:
        d = boot_deltas(best["clust"], base["clust"], seed=42)
        dd = d["cit_p"]
        if sig(dd):
            gain = f"+{dd[0]*100:.1f} pt，95% CI [{dd[1]*100:+.1f}, {dd[2]*100:+.1f}]（CI 不跨 0）"
        else:
            gain = (f"**无显著差异**：趋势 {dd[0]*100:+.1f} pt，95% CI [{dd[1]*100:+.1f}, {dd[2]*100:+.1f}]（跨 0）")
        lines.append(f"| 4 | 增益与 95% CI | {gain} | delta 表 Δcit_p 行 |")
        cost_chg = (best["agg"]["cost_cny"] - base["agg"]["cost_cny"]) / base["agg"]["cost_cny"] * 100 if base["agg"]["cost_cny"] else 0
        lines.append(f"| 5 | 单次成本变化 | {cost_chg:+.1f}% | 主表 cost_cny 之比 |")
        fb_best, tot_best = failure_counts(best["clust"])
        fb_base, tot_base = failure_counts(base["clust"]) if base else ({}, 0)
        lines.append(
            f"| 6 | 失败四分类 | best：{share_txt(fb_best, tot_best)}；base：{share_txt(fb_base, tot_base)} | 见第 3 节 |"
        )
    lines.append("")

    # 第 4 项若 CI 跨 0，必须给一条真实成立的替代结论
    if base and best_combo != base_combo:
        alt = []
        for combo in combos:
            if combo not in data or combo == base_combo:
                continue
            d = boot_deltas(data[combo]["clust"], base["clust"], seed=42)
            if sig(d["cost_cny"]) and d["cost_cny"][0] < 0:
                chg = d["cost_cny"][0] / base["agg"]["cost_cny"] * 100
                lo = d["cost_cny"][1] / base["agg"]["cost_cny"] * 100
                hi = d["cost_cny"][2] / base["agg"]["cost_cny"] * 100
                alt.append(
                    f"- `{common.label_of(*combo)}`：单任务成本 {chg:+.1f}%（95% CI [{lo:+.1f}%, {hi:+.1f}%]，不跨 0），"
                    f"同时 success {d['success'][0]*100:+.1f} pt（CI [{d['success'][1]*100:+.1f}, {d['success'][2]*100:+.1f}]，跨 0）。"
                )
                break
        if alt:
            lines.append("**可替代的真实结论（CI 不跨 0）**：\n")
            lines.extend(alt)
            lines.append("")

    out = Path(args.out) if args.out else common.RESULTS / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
