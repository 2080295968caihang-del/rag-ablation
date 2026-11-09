"""离线复查（¥0）：只读现有 jsonl，补 5 个「没有数据支撑」的点。

覆盖 EXPERIMENT_PLAN.md 的 E0-2 ~ E0-6：
  E0-2 success 同时要求引用召回（cit_r >= 0.5）后结论是否还成立
  E0-3 无引用约束的纯答案 F1 / EM（可与论文数字对话）
  E0-4 「一步收口」到底是任务简单还是系统够强
  E0-5 峰时 / 谷时成本区间
  E0-6 失败四分类的样本量是否够（Wilson 区间）

用法：
  python eval/offline_checks.py
产物：eval/results_topk/offline_checks.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import string
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import metrics as M  # noqa: E402
from eval import common  # noqa: E402

N_BOOT = 2000
TAG = "hybrid_rr-on_rw-off"
CLASSES = ("未召回", "召回未读入", "读入推理错", "引用错配")
SAMPLES: dict[str, dict] = {}


def rng_for(key: str, qs: list[str]) -> np.random.Generator:
    h = hashlib.sha256(("|".join([key, *qs])).encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(h, "big"))


def norm_ans(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_score(pred: str, gold: str) -> float:
    return 1.0 if norm_ans(pred) == norm_ans(gold) else 0.0


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - r) / d, (c + r) / d


def load(d: Path, k: int | None, seeds):
    """逐记录读，返回 {qid: {字段: [每 seed 的值]}}。"""
    per: dict[str, dict[str, list]] = {}
    for sd in seeds:
        p = d / (f"{TAG}_top{k}_seed{sd}.jsonl" if k else f"{TAG}_seed{sd}.jsonl")
        if not p.exists():
            raise SystemExit(f"缺少 {p}")
        for r in common.load_records(p):
            s = SAMPLES.get(r["qid"])
            if s is None:
                continue
            f1 = M.f1(r["answer"], s["answer"])
            cp, cr = M.citation_scores(r["citations"], r["context_pids"], s["gold_pids"])
            gold = set(s["gold_pids"])
            ctx = set(r.get("context_pids") or [])
            slot = per.setdefault(r["qid"], {"f1": [], "cp": [], "cr": [], "cov": [], "tc": [],
                                             "cost": [], "pt": [], "ct": [], "em": [],
                                             "cls": [], "ok": [], "ok3": []})
            slot["f1"].append(f1)
            slot["cp"].append(cp)
            slot["cr"].append(cr)
            slot["cov"].append(len(gold & ctx) / len(gold) if gold else 0.0)
            slot["tc"].append(r.get("tool_calls", 0))
            slot["cost"].append(r["cost_cny"])
            slot["pt"].append(r["prompt_tokens"])
            slot["ct"].append(r["completion_tokens"])
            slot["em"].append(em_score(r["answer"], s["answer"]))
            slot["cls"].append(M.failure_class(r, s["gold_pids"], f1, cp))
            slot["ok"].append(M.is_success(f1, cp))
            slot["ok3"].append(1.0 if (f1 >= 0.5 and cp >= 0.5 and cr >= 0.5) else 0.0)
    return per


def agg(per: dict, key: str) -> dict[str, float]:
    return {q: float(np.mean(v[key])) for q, v in per.items()}


def paired(a: dict, b: dict, qs: list[str], key: str):
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
    ap.add_argument("--holdout-dir", default=str(common.ROOT / "eval" / "results_holdout"))
    ap.add_argument("--strict-thr", type=float, default=0.5)
    ap.add_argument("--out", default=str(common.ROOT / "eval" / "results_topk" / "offline_checks.md"))
    args = ap.parse_args()

    global SAMPLES
    SAMPLES = {s["qid"]: s for s in common.load_samples(Path(args.data))}
    seeds = common.seeds_in() or list(common.SEEDS)

    p5 = load(Path(args.base_dir), None, seeds)
    p20 = load(Path(args.topk_dir), 20, seeds)
    qs = sorted(set(p5) & set(p20))
    T = args.strict_thr
    L = ["# 离线复查（¥0）：补 5 个没有数据支撑的点\n",
         f"数据：现有 jsonl，`{TAG}`，{len(qs)} 题 × {len(seeds)} seed，qid 聚类配对（bootstrap 2000 次，确定性播种）。\n"]

    # ---- E0-2 success 三条件 ----
    L.append("## E0-2 success 是否该同时要求引用召回（`cit_r >= 0.5`）\n")
    a2 = {q: 1.0 if (np.mean(p5[q]["f1"]) >= T and np.mean(p5[q]["cp"]) >= T and np.mean(p5[q]["cr"]) >= T) else 0.0 for q in qs}
    b2 = {q: 1.0 if (np.mean(p20[q]["f1"]) >= T and np.mean(p20[q]["cp"]) >= T and np.mean(p20[q]["cr"]) >= T) else 0.0 for q in qs}
    m5, m20, mu2, lo2, hi2 = paired(a2, b2, qs, "paired:succ:citr")
    base5 = float(np.mean([np.mean(p5[q]["ok"]) for q in qs]))
    base20 = float(np.mean([np.mean(p20[q]["ok"]) for q in qs]))
    L.append("| 口径 | k=5 success | k=20 success | Δ | 95% CI | 跨 0 |")
    L.append("|---|---|---|---|---|---|")
    L.append(f"| 二条件（现行，`f1` + `cit_p`） | {base5:.4f} | {base20:.4f} | "
             f"{(base20-base5)*100:+.2f} pt | — | — |")
    L.append(f"| 三条件（+ `cit_r >= {T}`） | {m5:.4f} | {m20:.4f} | {mu2*100:+.2f} pt | "
             f"[{lo2*100:+.2f}, {hi2*100:+.2f}] | {'是' if lo2 <= 0 <= hi2 else '**否**'} |")
    L.append("")

    # ---- E0-3 纯 F1 / EM ----
    f1_5, f1_20 = agg(p5, "f1"), agg(p20, "f1")
    em_5, em_20 = agg(p5, "em"), agg(p20, "em")
    L.append("## E0-3 无引用约束的纯答案指标（可与论文数字对话）\n")
    L.append("| 指标 | k=5 | k=20 | Δ | 95% CI |")
    L.append("|---|---|---|---|---|")
    for name, a, b in (("F1", f1_5, f1_20), ("EM", em_5, em_20)):
        _, _, mu, lo, hi = paired(a, b, qs, f"paired:{name.lower()}")
        L.append(f"| {name} | {np.mean([a[q] for q in qs]):.4f} | {np.mean([b[q] for q in qs]):.4f} | "
                 f"{mu*100:+.2f} pt | [{lo*100:+.2f}, {hi*100:+.2f}] |")
    L.append("\n_EM 用 SQuAD 式归一化（小写 / 去标点 / 去冠词）。这是 HotpotQA 论文口径，可与公开数字对比；\n"
             "但仍是 dev(validation) 分片、单模型，不能当成 SOTA 排名依据。_\n")

    # ---- E0-4 步数分组 ----
    L.append("## E0-4 「一步收口」是任务简单还是系统够强（k=5 档）\n")
    one = [q for q in qs if np.mean(p5[q]["tc"]) <= 1.0]
    multi = [q for q in qs if np.mean(p5[q]["tc"]) > 1.0]
    L.append(f"一步收口 {len(one)} 题（{len(one)/len(qs)*100:.1f}%），多步 {len(multi)} 题（{len(multi)/len(qs)*100:.1f}%）。\n")
    L.append("_口径：按 qid 对两 seed 的 `tool_calls` 取均值后分组；README 的 78.4% 是按**记录**统计"
             "（1,200 条里 941 条 1 步），差异来自分母，不是矛盾。_\n")
    L.append("| 分组 | 题数 | gold 覆盖率 | success | 平均 gold 条数 | 平均步数 |")
    L.append("|---|---|---|---|---|---|")
    for name, g in (("一步收口", one), ("≥2 步", multi)):
        if not g:
            continue
        cov = np.mean([np.mean(p5[q]["cov"]) for q in g])
        succ = np.mean([np.mean(p5[q]["ok"]) for q in g])
        ng = np.mean([len(SAMPLES[q]["gold_pids"]) for q in g])
        tc = np.mean([np.mean(p5[q]["tc"]) for q in g])
        L.append(f"| {name} | {len(g)} | {cov*100:.1f}% | {succ:.3f} | {ng:.2f} | {tc:.2f} |")
    L.append("")

    # ---- E0-5 峰谷成本 ----
    pt = float(np.mean([np.mean(p5[q]["pt"]) for q in qs]))
    ct = float(np.mean([np.mean(p5[q]["ct"]) for q in qs]))
    pt20 = float(np.mean([np.mean(p20[q]["pt"]) for q in qs]))
    ct20 = float(np.mean([np.mean(p20[q]["ct"]) for q in qs]))
    pin, pout, fx = 0.15, 0.60, 7.1
    def cny(p_tok, c_tok, mult=1.0):
        return (p_tok / 1e6 * pin * mult + c_tok / 1e6 * pout * mult) * fx
    L.append("## E0-5 成本区间（off-peak 账单价 vs 峰时翻倍）\n")
    L.append("| 配置 | 平均 prompt tok | 平均 completion tok | off-peak（记账） | 峰时（×2） |")
    L.append("|---|---|---|---|---|")
    for name, a, b in (("k=5", pt, ct), ("k=20", pt20, ct20)):
        L.append(f"| {name} | {a:.0f} | {b:.0f} | ¥{cny(a,b):.5f} | ¥{cny(a,b,2):.5f} |")
    L.append(f"\n口径：输入 ${pin}/1M、输出 ${pout}/1M、汇率 {fx}（`app/llm.py` 常量）。"
             "「峰时 ×2」按 README 记录的峰时翻倍规则推算，**未经账单验证**，只用于给区间。\n")

    # ---- E0-6 分类样本量 ----
    L.append("## E0-6 失败四分类的样本量（含 Wilson 95% 区间）\n")
    L.append(f"_口径：本表分母是「600 题 × {len(seeds)} seed」的配对子集；`confirm.md` §4 的 929/153 是"
             "「1500 题 × 2 seed」全量记录的计数，两者分母不同，比例可比、绝对数不可直接比。_\n")
    L.append("| 档位 | 类型 | 记录数 | 占比 | 95% Wilson |")
    L.append("|---|---|---|---|---|")
    for label, per in (("k=5", p5), ("k=20", p20)):
        cnt = {c: 0 for c in CLASSES}
        for v in per.values():
            for c in v["cls"]:
                if c in cnt:
                    cnt[c] += 1
        tot = sum(cnt.values())
        for c in CLASSES:
            lo, hi = wilson(cnt[c], tot)
            flag = " ⚠样本不足" if cnt[c] < 30 else ""
            L.append(f"| {label} | {c} | {cnt[c]} | {cnt[c]/max(1,tot)*100:.0f}% | "
                     f"[{lo*100:.1f}%, {hi*100:.1f}%]{flag} |")
    L.append("\n判读：<30 条的类别不下结论（Wilson 区间宽度 >±15 pt）。\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[ok] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
