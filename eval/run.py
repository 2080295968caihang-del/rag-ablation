"""跑单组单 seed。用法见 HANDOVER 11。

  python eval/run.py --mock --limit 3
  python eval/run.py --retriever dense --rerank off --rewrite off --seed 42
  python eval/run.py --gold --limit 100           # gold 注入对照
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import common  # noqa: E402

try:
    from dotenv import load_dotenv

    # override=True：让项目内 .env 覆盖 shell/系统里的同名变量，
    # 否则外部注入的旧 DEEPSEEK_API_KEY 会一直生效，改 .env 看起来“没反应”。
    load_dotenv(common.ROOT / ".env", override=True)
except Exception:  # noqa: BLE001
    pass

from app import agent, llm, retrievers  # noqa: E402


def _load_done(path: Path) -> tuple[set[str], list[str], int]:
    """读已有结果：返回 (已完成 qid, 有效行, 坏行数)。

    逐行 flush + 中断只会砍在行边界，理论上不会有坏行；仍然容错，
    避免一次手滑编辑让整份结果作废。
    """
    if not path.exists():
        return set(), [], 0
    done: set[str] = set()
    good: list[str] = []
    bad = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                done.add(json.loads(s)["qid"])
                good.append(s)
            except Exception:  # noqa: BLE001
                bad += 1
    return done, good, bad


def _stop_requested() -> bool:
    """项目根目录出现 STOP 文件即请求暂停（新一批任务开始前检查）。"""
    return (common.ROOT / "STOP").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="dense", choices=list(common.RETRIEVERS))
    ap.add_argument("--rerank", default="off", choices=["on", "off"])
    ap.add_argument("--rewrite", default="off", choices=["on", "off"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--gold", action="store_true", help="gold 段落直接注入上下文（归因对照）")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--data", default=str(common.DATA))
    ap.add_argument("--out", default="")
    ap.add_argument("--no-resume", action="store_true",
                    help="忽略已有结果文件，从零重跑（默认会跳过已完成的 qid 续跑）")
    args = ap.parse_args()

    rr, rw = args.rerank == "on", args.rewrite == "on"
    tag = common.tag_of(args.retriever, rr, rw, args.gold)
    out = Path(args.out) if args.out else common.result_path(tag, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)

    samples = common.load_samples(Path(args.data))
    if args.limit:
        samples = samples[: args.limit]

    cfg = {
        "retriever": args.retriever,
        "rerank": rr,
        "rewrite": rw,
        "top_k": args.top_k,
        "max_steps": args.max_steps,
        "mock": args.mock,
        "gold_inject": args.gold,
        "seed": args.seed,
    }

    if not args.mock:
        st = retrievers.warmup()
        print(f"[models] {st}")
        # 静默降级会污染实验（dense 变空、rerank 变原序），所以这里直接中止。
        if not st.get("embedding_ok"):
            print(f"[abort] 嵌入模型加载失败，dense/hybrid 会全空，不跑：{st.get('embedding_error')}")
            return 3
        if args.rerank == "on" and not st.get("rerank_ok"):
            print(f"[abort] 重排模型加载失败，rerank=on 会退化为原序，不跑：{st.get('rerank_error')}")
            return 3
        ok, msg = llm.preflight()
        if not ok:
            print(f"[abort] 预检失败（余额/鉴权/网络），未消耗任何任务：{msg}")
            return 3

    llm.ledger_reset()
    t0 = time.time()
    done = 0
    total = len(samples)

    # ---- 续跑：读已有结果，只补缺的 qid ----
    done_qids: set[str] = set()
    n_existing = 0
    if not args.no_resume:
        done_qids, good, bad = _load_done(out)
        n_existing = len(good)
        if bad:
            bak = out.with_name(out.name + ".bad")
            bak.write_text("\n".join(good) + "\n", encoding="utf-8")
            out.write_text("\n".join(good) + "\n", encoding="utf-8")
            print(f"[repair] {out.name} 有 {bad} 行损坏，已剔除并备份到 {bak.name}")
        if n_existing:
            print(f"[resume] {out.name} 已有 {n_existing} 条，跳过已完成 qid")

    todo = [s for s in samples if s["qid"] not in done_qids]
    if not todo:
        print(f"[skip] {out.name} 已完整（{n_existing}/{total}），无事可做")
        return 0
    if n_existing:
        print(f"[resume] 本次补跑 {len(todo)} 条")

    def work(s):
        return agent.run_task(s, cfg)

    def emit(rec):
        nonlocal done
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        done += 1
        print(f"[{n_existing+done}/{total}] {rec['qid']} {rec['terminated_by']} "
              f"ans={rec['answer'][:40]!r} calls={rec['tool_calls']}", flush=True)

    paused = False
    fh = out.open("a" if n_existing else "w", encoding="utf-8")
    try:
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                it = iter(todo)
                while True:
                    batch = list(itertools.islice(it, args.workers))
                    if not batch:
                        break
                    for rec in ex.map(work, batch):
                        emit(rec)
                    # 只在批边界检查，暂停时本批结果已全部落盘，不浪费
                    if _stop_requested():
                        paused = True
                        print(f"[pause] 检测到 STOP，停在新一批边界（本批 {len(batch)} 条已落盘）")
                        break
        else:
            for s in todo:
                if _stop_requested():
                    paused = True
                    print("[pause] 检测到 STOP，已停在任务边界")
                    break
                emit(work(s))
    except llm.LLMError as e:
        fh.close()
        # 不再删半份结果：已完成的行都是有效观测，续跑从断点继续。
        print(f"[abort] API 级失败，保留 {n_existing+done}/{total} 条，续跑会从断点继续：{e}")
        return 3
    finally:
        fh.close()

    cum = 0.0
    with out.open(encoding="utf-8") as f2:
        for line in f2:
            try:
                cum += json.loads(line)["cost_cny"]
            except Exception:  # noqa: BLE001
                pass
    led = llm.ledger_snapshot()
    print(f"[{'paused' if paused else 'done'}] {out} 本次 n={done} 累计 {n_existing+done}/{total} "
          f"{time.time()-t0:.1f}s 本次 calls={led['calls']} tokens={led['prompt_tokens']}+{led['completion_tokens']} "
          f"本次 cost=¥{led['cost_cny']:.4f} 文件累计 cost=¥{cum:.4f}")
    return 4 if paused else 0


if __name__ == "__main__":
    sys.exit(main())
