"""多步 Agent 循环。评测信息绝不进 prompt（answer / gold_pids / supporting_facts 一律隔离）。"""
from __future__ import annotations

import json
import time

from app import llm, retrievers


def _fmt(context: list[dict]) -> str:
    return "\n".join(
        f"[{i+1}] ({c['title']}) {c['text']}" for i, c in enumerate(context)
    )


def _messages(question: str, context: list[dict], history: list[str]) -> list[dict]:
    sys = (
        "You are a careful multi-hop question answering agent. "
        "Use ONLY the numbered evidence passages. "
        "Reply with strict JSON and nothing else."
    )
    user = [
        f"Question: {question}",
        "",
        "Evidence:",
        _fmt(context),
    ]
    if history:
        user += ["", "Searches already performed:"] + [f"- {h}" for h in history]
    user += [
        "",
        'Output one strict JSON object: {"action": "answer" | "search", "answer": "short answer", '
        '"citations": [1, 3], "query": "next search query"}',
        "Rules:",
        '- citations are evidence numbers (1..N) you actually used; they must be non-empty when action="answer".',
        '- If the evidence is insufficient, use action="search" and put a focused English query in "query".',
        "- Keep answer short (entity or yes/no or a few words).",
    ]
    return [{"role": "system", "content": sys}, {"role": "user", "content": "\n".join(user)}]


def _parse(text: str):
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):] if "{" in t else t
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        d = json.loads(t[s: e + 1])
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(d, dict) or d.get("action") not in ("answer", "search"):
        return None
    return d


def _mock_decision(sample: dict) -> dict:
    """假响应：gold 句前 3 个词作答 + 第一个候选当 citation。不调任何 API。"""
    gold = [p for p in sample["passages"] if p.get("is_gold")]
    ans = " ".join((gold[0]["text"].split()[:3]) if gold else ["unknown"])
    return {"action": "answer", "answer": ans, "citations": [1], "query": ""}


def run_task(sample: dict, cfg: dict) -> dict:
    qid = sample["qid"]
    question = sample["question"]
    top_k = int(cfg.get("top_k", 5))
    max_steps = int(cfg.get("max_steps", 8))
    # 候选池大小：None=旧行为（rerank 取 20，否则取 top_k）；<=0=该题全部候选；>0=取 max(cand_k, top_k)。
    # 旧行为把 20 写死，导致 top_k=40 时上下文也只有 20 条 —— 那组「k=40」其实是无效对照。
    cand_k = cfg.get("cand_k")
    use_rr = bool(cfg.get("rerank"))
    use_rw = bool(cfg.get("rewrite"))
    mode = cfg.get("retriever", "dense")
    mock = bool(cfg.get("mock"))
    gold_inject = bool(cfg.get("gold_inject"))

    llm.task_reset()
    t0 = time.time()
    tool_calls = 0
    all_retrieved: list[int] = []
    context_pids: list[int] = []
    answer, citations = "", []
    terminated_by = "max_steps"
    parse_errors = 0
    history: list[str] = []

    if gold_inject:
        gold = [p for p in sample["passages"] if p.get("is_gold")][:top_k]
        context = [{"pid": p["pid"], "title": p["title"], "text": p["text"], "score": 1.0} for p in gold]
        context_pids = [c["pid"] for c in context]
        all_retrieved = list(context_pids)
        if mock:
            d = _mock_decision(sample)
        else:
            try:
                r = llm.chat(_messages(question, context, []))
                d = _parse(r["text"]) or {"action": "answer", "answer": "", "citations": []}
            except llm.LLMError:
                raise
            except Exception:  # noqa: BLE001
                d = {"action": "answer", "answer": "", "citations": []}
        answer = str(d.get("answer") or "")
        citations = list(d.get("citations") or [])
        terminated_by = "gold_inject"
    else:
        index = retrievers.build_index(sample["passages"])
        queries = retrievers.rewrite(question) if use_rw else [question]
        pool: dict[int, dict] = {}
        no_new = 0

        for step in range(max_steps):
            new = 0
            for q in queries:
                if cand_k is None:
                    k_ret = 20 if use_rr else top_k
                elif int(cand_k) <= 0:
                    k_ret = 10 ** 6  # retrieve 内部会 clamp 到该题候选总数
                else:
                    k_ret = max(int(cand_k), top_k)
                cands = retrievers.retrieve(index, q, k=k_ret, mode=mode)
                tool_calls += 1
                all_retrieved.extend(c["pid"] for c in cands)
                if use_rr:
                    cands = retrievers.rerank(q, cands, k=top_k)
                for c in cands:
                    if c["pid"] not in pool:
                        pool[c["pid"]] = c
                        new += 1
                    elif c["score"] > pool[c["pid"]]["score"]:
                        pool[c["pid"]] = c
            context = sorted(pool.values(), key=lambda x: -x["score"])[:top_k]
            context_pids = [c["pid"] for c in context]
            no_new = no_new + 1 if new == 0 else 0

            d = _mock_decision(sample) if mock else None
            if d is None:
                try:
                    r = llm.chat(_messages(question, context, history))
                    d = _parse(r["text"])
                    if d is None:  # 解析失败重试 1 次
                        parse_errors += 1
                        r = llm.chat(
                            _messages(question, context, history)
                            + [{"role": "user", "content": "That was not valid JSON. Repeat with strict JSON only."}]
                        )
                        d = _parse(r["text"])
                except llm.LLMError:
                    raise
                except Exception:  # noqa: BLE001
                    d = None
            if d is None:
                terminated_by = "parse_error"
                answer, citations = "", []
                break

            if d["action"] == "answer":
                answer = str(d.get("answer") or "")
                citations = list(d.get("citations") or [])
                terminated_by = "answer"
                break

            nq = str(d.get("query") or "").strip()
            if no_new >= 2:
                # 证据已挖尽：强制收口一次，避免空答案把"改写组"系统性压低（对四组不公平）
                terminated_by = "no_new_evidence"
                answer = str(d.get("answer") or "")
                citations = list(d.get("citations") or [])
                if not answer and not mock:
                    try:
                        r = llm.chat(
                            _messages(question, context, history)
                            + [{"role": "user", "content":
                                "The evidence is exhausted. Answer now with strict JSON "
                                '{"action":"answer","answer":"...","citations":[..]} using only the evidence above.'}]
                        )
                        d2 = _parse(r["text"]) or {}
                        answer = str(d2.get("answer") or "")
                        citations = list(d2.get("citations") or [])
                    except llm.LLMError:
                        raise
                    except Exception:  # noqa: BLE001
                        pass
                break
            queries = [nq or question]
            history.append(queries[0])

    # 步数耗尽仍未作答：强制收口一次。
    # 不加这一步时 max_steps=1 的对照会得到空答案，把「单步 vs 多步」的差异放大成假效应。
    if not mock and not gold_inject and not answer and terminated_by == "max_steps":
        try:
            r = llm.chat(
                _messages(question, context, history)
                + [{"role": "user", "content":
                    "The step budget is exhausted. Answer now with strict JSON "
                    '{"action":"answer","answer":"...","citations":[..]} using only the evidence above.'}]
            )
            d2 = _parse(r["text"]) or {}
            answer = str(d2.get("answer") or "")
            citations = list(d2.get("citations") or [])
            terminated_by = "forced_answer"
        except llm.LLMError:
            raise
        except Exception:  # noqa: BLE001
            pass

    latency_ms = int((time.time() - t0) * 1000)
    delta = llm.task_usage()
    return {
        "qid": qid,
        "cfg": {"retriever": mode, "rerank": use_rr, "rewrite": use_rw, "top_k": top_k,
                "cand_k": cand_k, "max_steps": max_steps, "mock": mock, "gold_inject": gold_inject},
        "seed": cfg.get("seed"),
        "answer": answer,
        "citations": citations,
        "context_pids": context_pids,
        "all_retrieved_pids": sorted(set(all_retrieved)),
        "tool_calls": tool_calls,
        "latency_ms": latency_ms,
        "cost_cny": round(delta["cost_cny"], 6),
        "prompt_tokens": delta["prompt_tokens"],
        "completion_tokens": delta["completion_tokens"],
        "terminated_by": terminated_by,
        "parse_errors": parse_errors,
    }
