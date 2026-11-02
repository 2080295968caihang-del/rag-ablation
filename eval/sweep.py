"""跑满 8 组 × 3 seed，并写 manifest.json。已完成的组合自动跳过（可 --force 重跑）。

  python eval/sweep.py --seeds 42,43,44
  python eval/sweep.py --seeds 42 --only dense,hybrid --workers 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import common  # noqa: E402


def n_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def write_manifest(seeds, tags, started_at, expect_per_file: int, dataset: str = "") -> dict:
    import datetime as dt

    from app import llm

    runs, total = [], 0.0
    t_start, t_end = None, None
    for tag in tags:
        for seed in seeds:
            p = common.result_path(tag, seed)
            recs = common.load_records(p)
            if not recs:
                continue
            cost = sum(r.get("cost_cny", 0.0) for r in recs)
            total += cost
            runs.append(
                {
                    "tag": tag,
                    "seed": seed,
                    "file": str(p.relative_to(common.ROOT)),
                    "n": len(recs),
                    "cost_cny": round(cost, 6),
                    "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in recs),
                    "completion_tokens": sum(r.get("completion_tokens", 0) for r in recs),
                }
            )
    manifest = {
        "project": "rag-ablation",
        "model": llm.MODEL,
        "served_model": "deepseek-flash",
        "base_url": llm.BASE_URL,
        "temperature": 0.0,
        "prices": {
            "price_in_usd_per_mtok": llm.PRICE_IN_USD_PER_M,
            "price_out_usd_per_mtok": llm.PRICE_OUT_USD_PER_M,
            "usd_cny": llm.USD_CNY,
            "note": "DeepSeek 官方价目表已只列 deepseek-flash / deepseek-v4-pro；deepseek-chat 为别名，"
                    "实测由 deepseek-flash 承接，按 Flash off-peak cache-miss 价计",
        },
        "seeds": list(seeds),
        "dataset": dataset,
        "expected_per_file": expect_per_file,
        "git_sha": common.git_sha(),
        "started_at": started_at,
        "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
        "total_cost_cny": round(total, 4),
        "total_tasks": sum(r["n"] for r in runs),
    }
    out = common.RESULTS / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--only", default="", help="逗号分隔，过滤 retriever：dense,hybrid")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gold", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--data", default=str(common.DATA))
    ap.add_argument("--results", default="", help="结果目录，默认 eval/results")
    args = ap.parse_args()

    common.set_results(args.results or (common.ROOT / "eval" / "results"))
    n_samples = len(common.load_samples(Path(args.data)))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    only = set(x.strip() for x in args.only.split(",") if x.strip())
    combos = [
        (r, rr, rw)
        for r in common.RETRIEVERS
        for rr in common.RERANKS
        for rw in common.REWRITES
        if not only or r in only
    ]
    expect = args.limit or n_samples
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    tags = [common.tag_of(r, rr, rw, args.gold) for r, rr, rw in combos]

    t0 = time.time()
    for r, rr, rw in combos:
        for seed in seeds:
            tag = common.tag_of(r, rr, rw, args.gold)
            # --limit 只用于冒烟：结果写 _smoke/，避免半份文件被后续全量当成已完成
            p = common.RESULTS / "_smoke" / f"{tag}_seed{seed}.jsonl" if args.limit else common.result_path(tag, seed)
            if not args.force and n_lines(p) >= expect:
                print(f"[skip] {tag} seed={seed} ({n_lines(p)} rows)")
                continue
            if args.force:
                p.unlink(missing_ok=True)
            cmd = [
                sys.executable, str(common.ROOT / "eval" / "run.py"),
                "--retriever", r, "--rerank", "on" if rr else "off",
                "--rewrite", "on" if rw else "off", "--seed", str(seed),
                "--workers", str(args.workers), "--data", args.data, "--out", str(p),
            ]
            if args.force:
                cmd += ["--no-resume"]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if args.gold:
                cmd += ["--gold"]
            print(f"[run] {tag} seed={seed}", flush=True)
            rc = subprocess.run(cmd, cwd=common.ROOT).returncode
            if rc == 4:
                print(f"[pause] {tag} seed={seed} 已按 STOP 暂停，sweep 停止；"
                      f"删掉 STOP 后重跑本命令即从断点继续", flush=True)
                return 4
            if rc != 0:
                print(f"[fail] {tag} seed={seed} rc={rc}", flush=True)
                return rc

    if args.limit:
        print(f"[smoke] 跳过 manifest 写入，结果在 {common.RESULTS / '_smoke'}")
        return 0
    m = write_manifest(seeds, tags, started_at, expect, dataset=str(Path(args.data)))
    print(f"[manifest] tasks={m['total_tasks']} cost=¥{m['total_cost_cny']} "
          f"elapsed={(time.time()-t0)/60:.1f}min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
