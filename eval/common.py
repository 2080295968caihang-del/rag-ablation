"""公共路径 / 命名 / 数据读取。不是新分层，只是避免 4 个脚本各写一遍。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "hotpot_dev_100.jsonl"
RESULTS = ROOT / "eval" / "results"

RETRIEVERS = ("dense", "hybrid")
RERANKS = (False, True)
REWRITES = (False, True)
SEEDS = (42, 43, 44)
BASE = ("dense", False, False)


def seeds_in(results: Path | None = None) -> list[int]:
    """从结果目录的文件名反推实际跑过哪些 seed，避免脚本写死 SEEDS。"""
    import re

    d = results or RESULTS
    pat = re.compile(r"_seed(\d+)\.jsonl$")
    found = {int(m.group(1)) for p in d.glob("*_seed*.jsonl") if (m := pat.search(p.name))}
    return sorted(found)

LONG_NAMES = ("success", "cit_p", "cit_r", "f1", "p95_ms", "cost_cny")


def tag_of(retriever: str, rerank: bool, rewrite: bool, gold: bool = False) -> str:
    t = f"{retriever}_{'rr-on' if rerank else 'rr-off'}_{'rw-on' if rewrite else 'rw-off'}"
    return t + "_goldinj" if gold else t


def label_of(retriever: str, rerank: bool, rewrite: bool) -> str:
    name = f"{retriever} / {'on' if rerank else 'off'} / {'on' if rewrite else 'off'}"
    return name + (" (base)" if (retriever, rerank, rewrite) == BASE else "")


def result_path(tag: str, seed: int) -> Path:
    return RESULTS / f"{tag}_seed{seed}.jsonl"


def set_results(p) -> Path:
    """切换结果目录（600 条扩样本实验写到 results_600/，不污染 100 条的结果）。"""
    global RESULTS
    RESULTS = Path(p)
    RESULTS.mkdir(parents=True, exist_ok=True)
    return RESULTS


def load_samples(path: Path = DATA) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_tag(tag: str):
    """'hybrid_rr-on_rw-off' -> (retriever, rerank, rewrite)"""
    parts = tag.split("_")
    return parts[0], parts[1] == "rr-on", parts[2] == "rw-on"


def git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip() or "no-git"
    except Exception:  # noqa: BLE001
        return "no-git"
