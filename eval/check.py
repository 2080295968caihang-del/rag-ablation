"""交付前断言。全绿才算完成。

  python eval/check.py                # 严格模式（DoD 全量）
  python eval/check.py --allow-partial # 管线自检：只校验已有文件，不要求 24 个
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import common  # noqa: E402

FAILS: list[str] = []
OKS: list[str] = []


def check(cond: bool, msg: str) -> None:
    (OKS if cond else FAILS).append(msg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    # ---- 1. 数据集 ----
    samples = common.load_samples()
    check(len(samples) == 100, f"数据集 100 条（实测 {len(samples)}）")
    qids = [s["qid"] for s in samples]
    check(len(set(qids)) == len(qids), "qid 无重复")
    bad_gold = [s["qid"] for s in samples if len(s["gold_pids"]) < 2]
    check(not bad_gold, f"每条 gold_pids>=2（违例 {len(bad_gold)}）")
    bad_pid = [s["qid"] for s in samples if [p["pid"] for p in s["passages"]] != list(range(len(s["passages"])))]
    check(not bad_pid, f"pid 为 0..n-1 连续（违例 {len(bad_pid)}）")
    bad_flag = [
        s["qid"] for s in samples
        if sorted(p["pid"] for p in s["passages"] if p["is_gold"]) != sorted(s["gold_pids"])
    ]
    check(not bad_flag, f"is_gold 与 gold_pids 一致（违例 {len(bad_flag)}）")

    # ---- 2. 结果文件 ----
    combos = [(r, rr, rw) for r in common.RETRIEVERS for rr in common.RERANKS for rw in common.REWRITES]
    need = [common.tag_of(*c) for c in combos]
    missing, wrong_n, wrong_qid = [], [], []
    for tag in need:
        for seed in common.SEEDS:
            p = common.result_path(tag, seed)
            recs = common.load_records(p)
            if not recs:
                missing.append(f"{tag}@{seed}")
                continue
            if len(recs) != 100:
                wrong_n.append(f"{tag}@{seed}:{len(recs)}")
            if sorted(r["qid"] for r in recs) != sorted(qids):
                wrong_qid.append(f"{tag}@{seed}")
    if args.allow_partial:
        check(True, f"[partial] 已有 {len(need)*len(common.SEEDS)-len(missing)}/{len(need)*len(common.SEEDS)} 个结果文件")
    else:
        check(not missing, f"24 个结果文件齐（缺 {missing[:3]} 共 {len(missing)}）")
        check(not wrong_n, f"每文件 100 行（异常 {wrong_n[:3]}）")
        check(not wrong_qid, f"qid 与数据集一致（异常 {wrong_qid[:3]}）")

    # ---- 3. manifest ----
    mp = common.RESULTS / "manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        for k in ("model", "prices", "seeds", "git_sha", "started_at", "finished_at", "total_cost_cny"):
            check(k in m, f"manifest 含 {k}")
        check(m.get("seeds") == list(common.SEEDS), f"manifest seeds={m.get('seeds')}")
        check(float(m.get("total_cost_cny", 0)) > 0, f"manifest 总成本>0（{m.get('total_cost_cny')}）")
        if not args.allow_partial:
            check(m.get("total_tasks") == 2400, f"manifest total_tasks=2400（{m.get('total_tasks')}）")
    else:
        check(False, "manifest.json 存在（缺失）")

    # ---- 4. summary ----
    sp = common.RESULTS / "summary.md"
    if sp.exists():
        txt = sp.read_text(encoding="utf-8")
        check("| config | success | cit_p | cit_r | f1 | p95_ms | cost_cny |" in txt, "summary 主表表头")
        check("Δsuccess vs base [95%CI]" in txt, "summary 有 Δsuccess CI 列")
        check("〔" not in txt and "〕" not in txt, "summary 无未填占位符")
        check("未召回" in txt and "引用错配" in txt, "summary 有失败四分类")
        check("gold 注入" in txt, "summary 有 gold 注入对照")
        need_ci = 1 if args.allow_partial else 7
        check(txt.count(" pt [") >= need_ci, f"delta 行带 CI（{txt.count(' pt [')} 处，需 ≥{need_ci}）")
    else:
        check(args.allow_partial, "summary.md 存在")

    # ---- 5. 指标口径复算 ----
    from app import metrics

    check(abs(metrics.f1("yes", "yes") - 1.0) < 1e-9, "f1 自反=1")
    check(abs(metrics.f1("no", "yes")) < 1e-9, "f1 无关=0")
    p, r = metrics.citation_scores([1, 3], [10, 20, 30], [10, 30])
    check(abs(p - 1.0) < 1e-9 and abs(r - 1.0) < 1e-9, "citation 命中全对")
    p2, r2 = metrics.citation_scores([9], [10, 20], [10, 20])
    check(p2 == 0.0 and r2 == 0.0, "越界 citation 记为错配")
    check(metrics.is_success(0.5, 0.5) == 1 and metrics.is_success(0.49, 1.0) == 0, "success 阈值")
    mean, lo, hi = metrics.bootstrap_ci([1, 1, 1], n=200, seed=0)
    check(lo <= mean <= hi and abs(mean - 1) < 1e-9, "bootstrap 边界")

    for m_ in OKS:
        print(f"  ok   {m_}")
    for m_ in FAILS:
        print(f"  FAIL {m_}")
    print(f"\n{len(OKS)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
