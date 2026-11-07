"""交付前断言。全绿才算完成。

  python eval/check.py                # 严格模式（DoD 全量）
  python eval/check.py --allow-partial # 管线自检：只校验已有文件，不要求 24 个
"""
from __future__ import annotations

import argparse
import json
import re
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
    ap.add_argument("--data", default=str(common.DATA))
    ap.add_argument("--results", default="")
    args = ap.parse_args()
    common.set_results(args.results or common.RESULTS)

    # ---- 1. 数据集 ----
    samples = common.load_samples(Path(args.data))
    n_s = len(samples)
    check(n_s >= 100, f"数据集条数 {n_s}（≥100）")
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
    seeds = common.seeds_in() or list(common.SEEDS)
    combos = [(r, rr, rw) for r in common.RETRIEVERS for rr in common.RERANKS for rw in common.REWRITES]
    need = [common.tag_of(*c) for c in combos]
    missing, wrong_n, wrong_qid = [], [], []
    for tag in need:
        for seed in seeds:
            p = common.result_path(tag, seed)
            recs = common.load_records(p)
            if not recs:
                missing.append(f"{tag}@{seed}")
                continue
            if len(recs) != n_s:
                wrong_n.append(f"{tag}@{seed}:{len(recs)}")
            if sorted(r["qid"] for r in recs) != sorted(qids):
                wrong_qid.append(f"{tag}@{seed}")
    if args.allow_partial:
        check(True, f"[partial] 已有 {len(need)*len(seeds)-len(missing)}/{len(need)*len(seeds)} 个结果文件")
    else:
        check(not missing, f"{len(need)*len(seeds)} 个结果文件齐（缺 {missing[:3]} 共 {len(missing)}）")
        check(not wrong_n, f"每文件 {n_s} 行（异常 {wrong_n[:3]}）")
        check(not wrong_qid, f"qid 与数据集一致（异常 {wrong_qid[:3]}）")

    # 完整性：不许有 parse_error / api_error（402 曾被当成 parse_error 吞掉，污染过一整批）
    bad_term = []
    for p in sorted(common.RESULTS.glob("*_seed*.jsonl")):
        for r in common.load_records(p):
            if r.get("terminated_by") in ("parse_error", "api_error"):
                bad_term.append(f"{p.name}:{r.get('qid')}")
                break
    check(not bad_term, f"结果无 parse_error/api_error（异常 {bad_term[:3]} 共 {len(bad_term)}）")

    # ---- 3. manifest ----
    mp = common.RESULTS / "manifest.json"
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        for k in ("model", "prices", "seeds", "git_sha", "started_at", "finished_at", "total_cost_cny"):
            check(k in m, f"manifest 含 {k}")
        check(sorted(m.get("seeds") or []) == seeds, f"manifest seeds={m.get('seeds')} 与文件名一致 {seeds}")
        check(float(m.get("total_cost_cny", 0)) > 0, f"manifest 总成本>0（{m.get('total_cost_cny')}）")
        if not args.allow_partial:
            want = 8 * len(seeds) * n_s
            # total_tasks 可含 gold 注入等附加 run，用 sweep_tasks 校验 8 组消融本身
            got = m.get("sweep_tasks", m.get("total_tasks"))
            check(got == want, f"manifest 消融任务数={want}（{got}）")
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
        check("工具调用步数分布" in txt, "summary 有 tool_calls 步数分布")
        check("后续阶段" in txt, "summary 指向 confirm.md（避免两份结论打架）")
    else:
        check(args.allow_partial, "summary.md 存在")

    # ---- 4b. 确认性检验 + top_k 干预（第二阶段）----
    cp600 = common.ROOT / "data" / "hotpot_dev_600.jsonl"
    conf = common.ROOT / "eval" / "results_confirm_1500" / "confirm.md"
    if conf.exists():
        ct = conf.read_text(encoding="utf-8")
        check("留出集（未参与选组）" in ct, "confirm 有留出集行")
        check("有效对照" in ct and "候选池约束" in ct, "confirm 有槽位有效性与候选池约束说明")
        check("失败类型转移" in ct, "confirm 有失败类型转移（机制证据）")
        check("单步 vs 多步" in ct, "confirm 有单步/多步对照")
        check(ct.count(" pt ") >= 10, f"confirm 带 CI 行数（{ct.count(' pt ')} 处）")
        tkd = common.ROOT / "eval" / "results_topk"
        if tkd.exists():
            n600 = len(common.load_samples(cp600)) if cp600.exists() else 0
            bad = []
            found = set()
            for p in sorted(tkd.glob("*_top*_seed*.jsonl")):
                recs = common.load_records(p)
                if n600 and len(recs) != n600:
                    bad.append(f"{p.name}:{len(recs)}/{n600}")
                if any(r.get("terminated_by") in ("parse_error", "api_error") for r in recs):
                    bad.append(f"{p.name}:bad_term")
                m = re.search(r"_top(\d+)_seed", p.name)
                if m:
                    found.add(int(m.group(1)))
            check(not bad, f"top_k 结果均为 {n600} 行且无坏终止（异常 {bad[:3]} 共 {len(bad)}）")
            for k in sorted(found):
                check(f"top_k={k}" in ct, f"confirm 含 top_k={k} 段")
        # 留出集复现：行数、配置、坏终止，且 confirm 必须有该节
        hod = common.ROOT / "eval" / "results_holdout"
        if hod.exists() and any(hod.glob("*.jsonl")):
            nho = len(common.load_samples(common.ROOT / "data" / "hotpot_dev_holdout900.jsonl"))
            bad_ho = []
            for p_ in sorted(hod.glob("*_top*_seed*.jsonl")):
                recs = common.load_records(p_)
                if len(recs) != nho:
                    bad_ho.append(f"{p_.name}:{len(recs)}/{nho}")
                if any(r.get("terminated_by") in ("parse_error", "api_error") for r in recs):
                    bad_ho.append(f"{p_.name}:bad_term")
                if any((r.get("cfg") or {}).get("top_k") != 20 for r in recs):
                    bad_ho.append(f"{p_.name}:cfg_topk!=20")
            check(not bad_ho, f"留出集 {nho} 行/config=20/无坏终止（异常 {bad_ho[:3]} 共 {len(bad_ho)}）")
            check("留出 900 题" in ct, "confirm 含留出集复现节")
            check("复现检查" in ct, "confirm 含 600 vs 900 复现对照表")
    else:
        check(args.allow_partial, "confirm.md 存在")

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
