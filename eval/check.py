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
            check("重复运行噪声" in ct, "confirm 含同配置重跑噪声节")
        # 同批 k=5 基线（与 k=20 同一天跑），必须完整且配置正确
        hbd = common.ROOT / "eval" / "results_holdout_fresh"
        if hbd.exists() and any(hbd.glob("*.jsonl")):
            nho = len(common.load_samples(common.ROOT / "data" / "hotpot_dev_holdout900.jsonl"))
            bad_f = []
            for p_ in sorted(hbd.glob("*_seed*.jsonl")):
                recs = common.load_records(p_)
                if len(recs) != nho:
                    bad_f.append(f"{p_.name}:{len(recs)}/{nho}")
                if any((r.get("cfg") or {}).get("top_k") != 5 for r in recs):
                    bad_f.append(f"{p_.name}:cfg_topk!=5")
                if any(r.get("terminated_by") in ("parse_error", "api_error") for r in recs):
                    bad_f.append(f"{p_.name}:bad_term")
            check(not bad_f, f"同批 k=5 基线 {nho} 行/config=5/无坏终止（异常 {bad_f[:3]} 共 {len(bad_f)}）")
        # 阶段 2 的前 600 题是阶段 1 结果的复制（记账只算一次，README 有注）
        c1500 = common.ROOT / "eval" / "results_confirm_1500"
        r600 = common.ROOT / "eval" / "results_600"
        if c1500.exists() and r600.exists():
            mism = 0
            for tag in ("dense_rr-off_rw-off", "hybrid_rr-on_rw-off"):
                for sd in (42, 43):
                    a_ = r600 / f"{tag}_seed{sd}.jsonl"
                    b_ = c1500 / f"{tag}_seed{sd}.jsonl"
                    if not (a_.exists() and b_.exists()):
                        continue
                    amap = {r["qid"]: r for r in common.load_records(a_)}
                    for r in common.load_records(b_):
                        if r["qid"] in amap and amap[r["qid"]] != r:
                            mism += 1
            check(mism == 0, f"阶段 2 的前 600 题确为阶段 1 复制（不一致 {mism} 条，须逐条相同）")
    else:
        check(args.allow_partial, "confirm.md 存在")

    # ---- 4.9 设计文档（数据 / 参数 / 技术栈 / 进度）----
    dz = common.ROOT / "DESIGN.md"
    check(dz.exists(), "DESIGN.md 存在")
    if dz.exists():
        dt = dz.read_text(encoding="utf-8")
        for sec in ("## 1. 数据", "## 2. 参数", "## 3. 技术栈", "## 4. 当前进度"):
            check(sec in dt, f"DESIGN 含「{sec[3:]}」节")
        # 总账不能手写：直接按「逐条 cost 求和 + 去重」现算，再要求两份文档都引用同一个数。
        # 去重规则：results_confirm_1500 里前 600 题的记录是阶段 1 的复制（同配置同 seed，逐条相同）。
        import json as _json
        f600 = {s["qid"] for s in common.load_samples(common.ROOT / "data" / "hotpot_dev_600.jsonl")}
        _n = _c = _dn = _dc = 0
        for _d in (common.ROOT / "eval").glob("results_*"):
            for _p in _d.glob("*.jsonl"):
                for _line in _p.open(encoding="utf-8"):
                    if not _line.strip():
                        continue
                    _r = _json.loads(_line)
                    _cost = float(_r.get("cost_cny") or 0)
                    _n += 1
                    _c += _cost
                    if _d.name == "results_confirm_1500" and _r["qid"] in f600:
                        _dn += 1
                        _dc += _cost
        want_n, want_c = f"{_n - _dn:,}", f"¥{_c - _dc:.2f}"
        rt = (common.ROOT / "README.md").read_text(encoding="utf-8")
        check(want_n in rt and want_c in rt, f"README 的总账与数据现算一致（{want_n} / {want_c}）")
        check(want_n in dt and want_c in dt, f"DESIGN 的总账与 README 一致（{want_n} / {want_c}）")
        # 未验证的东西必须自己承认，不许写成实测结论
        check("未实测" in dt or "未调参" in dt, "DESIGN 明确标注了未验证/未调参项")
        # 数据集规模：DESIGN 表里的段落总数必须与数据文件重算结果一致
        import json as _json
        for fname, want in (("hotpot_dev_100.jsonl", "3,883"),
                            ("hotpot_dev_600.jsonl", "24,519"),
                            ("hotpot_dev_1500.jsonl", "61,992"),
                            ("hotpot_dev_holdout900.jsonl", "37,473")):
            fp = common.ROOT / "data" / fname
            if not fp.exists():
                continue
            tot = sum(len(_json.loads(l)["passages"]) for l in fp.open(encoding="utf-8") if l.strip())
            check(f"{tot:,}" == want, f"DESIGN 中 {fname} 段落总数 {want} 与实测 {tot:,} 一致")

    # ---- 5. 跨文档数字一致：关键结论的 Δ 与 CI 必须与 confirm.md 现算值一致 ----
    cfp = common.ROOT / "eval" / "results_confirm_1500" / "confirm.md"
    docs = {n: (common.ROOT / n).read_text(encoding="utf-8")
            for n in ("README.md", "DESIGN.md", "PROJECT_SUMMARY.md")}
    if cfp.exists():
        cfl = cfp.read_text(encoding="utf-8").splitlines()

        def _cell(prefix: str, idx: int):
            for ln in cfl:
                if ln.startswith(prefix):
                    cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                    return cells[idx] if idx < len(cells) else None
            return None

        def _nums(cell: str) -> list[float]:
            return [float(x) for x in re.findall(r"[+-]\d+\.\d+", cell)]

        def _variants(cell: str) -> list[str]:
            ns = _nums(cell)
            out = []
            if len(ns) == 2:
                out += [f"[{ns[0]:+.2f},{ns[1]:+.2f}]", f"[{ns[0]:+.1f},{ns[1]:+.1f}]"]
            elif len(ns) == 1:
                out += [f"{ns[0]:+.2f}", f"{ns[0]:+.1f}"]
            return out

        def _flat(s: str) -> str:
            return re.sub(r"\s+", "", s)

        cases = [
            ("| 全集 | 1500 | success", 5, 6, "阶段2 全集1500 success", ("README.md", "DESIGN.md")),
            ("| 全集 | 1500 | cit_p", 5, 6, "阶段2 全集1500 cit_p", ("README.md", "DESIGN.md")),
            ("| **留出集（未参与选组）** | 900 | success", 5, 6, "阶段2 留出900 success",
             ("README.md", "DESIGN.md", "PROJECT_SUMMARY.md")),
            ("| success | 0.6944 | 0.7794 |", 3, 4, "阶段5 留出复现 success", ("README.md", "PROJECT_SUMMARY.md")),
        ]
        for pref, di, ci_i, label, need in cases:
            dcell, ccell = _cell(pref, di), _cell(pref, ci_i)
            if dcell is None or ccell is None:
                check(False, f"confirm.md 缺少 {label} 行，无法比对")
                continue
            ds, cs = _variants(dcell), _variants(ccell)
            for name in need:
                flat = _flat(docs[name])
                ok = any(_flat(v) in flat for v in ds) and any(_flat(v) in flat for v in cs)
                check(ok, f"{name} 引用的 {label} 与 confirm.md 一致（Δ {dcell} / CI {ccell}）")

        # 已作废的旧数字不得再出现在文档里
        stale = ("[+0.39, +3.78]", "[+0.67, +5.06]", "[+6.2, +10.8]", "69.7%",
                 "929→253", "153 条归零", "k=20 之后没测到")
        for name, txt in docs.items():
            flat = _flat(txt)
            bad = [s for s in stale if _flat(s) in flat]
            check(not bad, f"{name} 不含已作废的数字（发现 {bad}）")

        # ---- 5b. E1 追加实验：产物存在性 + 配置正确性 ----
        cross = common.ROOT / "eval" / "results_cross"
        bad_x = []
        for sd in (42, 43):
            p_ = cross / f"dense_rr-off_rw-off_top20_seed{sd}.jsonl"
            if not p_.exists():
                bad_x.append(f"{p_.name}:缺失")
                continue
            recs = common.load_records(p_)
            if len(recs) != n600:
                bad_x.append(f"{p_.name}:{len(recs)}/{n600}")
            for r in recs:
                c_ = r.get("cfg") or {}
                if c_.get("retriever") != "dense" or c_.get("top_k") != 20:
                    bad_x.append(f"{p_.name}:cfg")
                    break
        check(not bad_x, f"dense 交叉验证 1,200 条且配置正确（异常 {bad_x[:3]}）")

        # 每档 top_k 必须真把槽位开到位：cand_k 显式给值，且 >= top_k（或 0 = 全部候选）
        bad_c = []
        for k_, ck_ in ((10, None), (20, None), (30, 30), (40, None), (50, 50), (200, 0)):
            for sd in (42, 43):
                p_ = tkd / f"hybrid_rr-on_rw-off_top{k_}_seed{sd}.jsonl"
                if not p_.exists():
                    bad_c.append(f"top{k_}/seed{sd}:缺失")
                    continue
                for r in common.load_records(p_):
                    c_ = r.get("cfg") or {}
                    if c_.get("top_k") != k_ or c_.get("cand_k") != ck_:
                        bad_c.append(f"top{k_}/seed{sd}:cfg={c_.get('top_k')}/{c_.get('cand_k')}")
                        break
        check(not bad_c, f"top_k 各档的 cfg 与其档位一致（异常 {bad_c[:3]} 共 {len(bad_c)}）")

        # 饱和结论必须是报告算出来的，不是文档里手写的
        check("饱和点 = top_k=30 的上一档" in ct, "confirm 的饱和判据算出的拐点是 k=30 的上一档")
        check("检索环节的天花板" in ct, "confirm 记录「检索做到 100% 覆盖率时的 success 上限」")
        for nm, need in (("extra_arms.md", "第 3 个 seed"), ("metric_robust.md", "宽松"),
                         ("cite_audit.md", "引用错配"), ("failure_audit.md", "抽检"),
                         ("manual_audit.md", "脚本判定")):
            f_ = common.ROOT / "eval" / "results_topk" / nm
            check(f_.exists() and need in f_.read_text(encoding="utf-8"), f"{nm} 存在且含「{need}」")

        # 跨配置数字：README 与 cross_config.md 必须同源
        cc = common.ROOT / "eval" / "results_cross" / "cross_config.md"
        if cc.exists():
            cct = _flat(cc.read_text(encoding="utf-8"))
            row = next((l for l in cc.read_text(encoding="utf-8").splitlines()
                        if l.startswith("| dense / off / off")), "")
            # 该行有多个 Δ（覆盖率/成功率/成本），只有 success 那格加粗，取加粗的那个
            bold = re.findall(r"\*\*([+-]\d+\.\d+ pt)\*\*", row)
            nums = bold or re.findall(r"[+-]\d+\.\d+ pt", row)
            check(bool(nums), "cross_config 的 dense 行有加粗的 Δsucc")
            check(nums and _flat(nums[0]) in _flat(docs["README.md"]),
                  f"README 引用的 dense 交叉 Δsucc 与 cross_config 一致（{nums[0] if nums else '—'}）")

        # 阈值敏感性表里 T=0.50 那行必须与 confirm.md §3.2 主表逐位一致
        c_d, c_ci = _cell("| success | 0.6842 | 0.7892", 3), _cell("| success | 0.6842 | 0.7892", 4)
        sp = common.ROOT / "eval" / "results_topk" / "sensitivity.md"
        if sp.exists() and c_d and c_ci:
            sl = sp.read_text(encoding="utf-8").splitlines()
            s_row = next((l for l in sl if l.startswith("| 0.50 |")), "")
            sc = [x.strip() for x in s_row.strip().strip("|").split("|")] if s_row else []
            check(len(sc) == 6 and sc[3] == c_d and sc[4] == c_ci,
                  f"sensitivity.md 的 T=0.50 行与 confirm.md §3.2 一致（Δ {c_d} / CI {c_ci}）")
        else:
            check(False, "缺少 sensitivity.md 或 confirm.md §3.2 success 行")

        # 失败分类的并列 tie-break 必须确定性（set 迭代顺序受 PYTHONHASHSEED 影响）
        cps = (common.ROOT / "eval" / "confirm.py").read_text(encoding="utf-8")
        check("max(set(" not in cps, "confirm.py 分类 tie-break 不依赖 set 迭代顺序")

    # ---- 6. 指标口径复算 ----
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

    # ---- 7. 文档里写的断言条数 == 实际条数（本节每份文档一条，也计入总数）----
    names = list(docs)
    n_final = len(OKS) + len(FAILS) + len(names)
    for name in names:
        check(f"{n_final} 项" in docs[name] or f"{n_final}/{n_final}" in docs[name],
              f"{name} 的断言条数记为 {n_final}")

    for m_ in OKS:
        print(f"  ok   {m_}")
    for m_ in FAILS:
        print(f"  FAIL {m_}")
    print(f"\n{len(OKS)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
