"""跑单组单 seed。用法见 HANDOVER 11。

  python eval/run.py --mock --limit 3
  python eval/run.py --retriever dense --rerank off --rewrite off --seed 42
  python eval/run.py --gold --limit 100           # gold 注入对照
"""
from __future__ import annotations

import argparse
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

    def work(s):
        return agent.run_task(s, cfg)

    fh = out.open("w", encoding="utf-8")
    try:
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for rec in ex.map(work, samples):
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
                    print(f"[{done}/{len(samples)}] {rec['qid']} {rec['terminated_by']} "
                          f"ans={rec['answer'][:40]!r} calls={rec['tool_calls']}", flush=True)
        else:
            for s in samples:
                rec = work(s)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                print(f"[{done}/{len(samples)}] {rec['qid']} {rec['terminated_by']} "
                      f"ans={rec['answer'][:40]!r} calls={rec['tool_calls']}", flush=True)
    except llm.LLMError as e:
        fh.close()
        out.unlink(missing_ok=True)
        print(f"[abort] API 级失败，已删除半份结果 {out.name}，本组需重跑：{e}")
        return 3
    finally:
        fh.close()

    led = llm.ledger_snapshot()
    print(f"[done] {out}  n={done}  {time.time()-t0:.1f}s  calls={led['calls']}  "
          f"tokens={led['prompt_tokens']}+{led['completion_tokens']}  cost=¥{led['cost_cny']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
