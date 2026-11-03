"""DeepSeek 调用 + token 记账 + 指数退避重试。

价格口径（写死在这里，供 manifest 引用）：
  端点 https://api.deepseek.com ；请求名 deepseek-chat（实测由 deepseek-flash 承接）
  输入 $0.15 / 1M（cache miss，off-peak）  输出 $0.60 / 1M
  汇率 7.1 CNY/USD
"""
from __future__ import annotations

import os
import threading
import time

BASE_URL = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

PRICE_IN_USD_PER_M = float(os.getenv("PRICE_IN_USD_PER_M", "0.15"))
PRICE_OUT_USD_PER_M = float(os.getenv("PRICE_OUT_USD_PER_M", "0.60"))
USD_CNY = float(os.getenv("USD_CNY", "7.1"))

_LOCK = threading.Lock()
_CLIENT = None
_LEDGER = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "retries": 0, "failures": 0}


def _client():
    global _CLIENT
    with _LOCK:
        if _CLIENT is None:
            from openai import OpenAI

            key = os.getenv("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError("DEEPSEEK_API_KEY 未设置")
            _CLIENT = OpenAI(api_key=key, base_url=BASE_URL, timeout=120.0, max_retries=0)
        return _CLIENT


class LLMError(RuntimeError):
    """API 级失败（余额不足 / 鉴权 / 限流耗尽）。必须向上抛，不许被当成解析失败吞掉。"""

    def __init__(self, msg: str, status=None):
        super().__init__(msg)
        self.status = status


def _status_of(exc: Exception):
    s = getattr(exc, "status_code", None)
    if s is None:
        s = getattr(getattr(exc, "response", None), "status_code", None)
    return s


def _retryable(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name or "ratelimit" in name:
        return True
    status = _status_of(exc)
    if not isinstance(status, int):
        return False
    # 5xx 一律重试：520/521/522 等 Cloudflare 系列是上游瞬时故障，
    # 之前只白名单 500/502/503/504，导致 520 被当成致命错误整批中止。
    return status == 429 or status == 408 or 500 <= status <= 599


def cost_of(prompt_tokens: int, completion_tokens: int) -> float:
    usd = (prompt_tokens / 1e6) * PRICE_IN_USD_PER_M + (completion_tokens / 1e6) * PRICE_OUT_USD_PER_M
    return usd * USD_CNY


def ledger_snapshot() -> dict:
    with _LOCK:
        snap = dict(_LEDGER)
    snap["cost_cny"] = cost_of(snap["prompt_tokens"], snap["completion_tokens"])
    return snap


def ledger_reset() -> None:
    with _LOCK:
        for k in _LEDGER:
            _LEDGER[k] = 0


# ---- 任务级计量（线程局部）----
# 全局 _LEDGER 做差分在多线程下会互相串账（实测把 600 条任务的 prompt tokens
# 从 160k 记成 400k），成本比较因此失真。这里按线程累加，一个线程只跑一条任务。
_TLS = threading.local()


def task_reset() -> None:
    _TLS.prompt_tokens = 0
    _TLS.completion_tokens = 0
    _TLS.calls = 0


def task_usage() -> dict:
    pt = getattr(_TLS, "prompt_tokens", 0)
    ct = getattr(_TLS, "completion_tokens", 0)
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "calls": getattr(_TLS, "calls", 0),
        "cost_cny": cost_of(pt, ct),
    }


def _task_add(prompt_tokens: int, completion_tokens: int) -> None:
    _TLS.prompt_tokens = getattr(_TLS, "prompt_tokens", 0) + prompt_tokens
    _TLS.completion_tokens = getattr(_TLS, "completion_tokens", 0) + completion_tokens
    _TLS.calls = getattr(_TLS, "calls", 0) + 1


def ledger_delta(before: dict, after: dict | None = None) -> dict:
    after = after or ledger_snapshot()
    d = {k: after[k] - before.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "calls", "retries", "failures")}
    d["cost_cny"] = after["cost_cny"] - before.get("cost_cny", 0.0)
    return d


def chat(messages, temperature: float = 0.0, max_retries: int = 5) -> dict:
    """单次对话。返回 {text, prompt_tokens, completion_tokens, latency_ms}。失败抛异常由调用方处理。"""
    delays = [1.0, 2.0, 4.0, 8.0, 16.0]
    last = None
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = _client().chat.completions.create(
                model=MODEL, messages=messages, temperature=temperature
            )
            latency = int((time.time() - t0) * 1000)
            usage = getattr(resp, "usage", None)
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            text = (resp.choices[0].message.content or "").strip()
            with _LOCK:
                _LEDGER["prompt_tokens"] += pt
                _LEDGER["completion_tokens"] += ct
                _LEDGER["calls"] += 1
                if attempt:
                    _LEDGER["retries"] += attempt
            _task_add(pt, ct)
            return {"text": text, "prompt_tokens": pt, "completion_tokens": ct, "latency_ms": latency}
        except Exception as e:  # noqa: BLE001
            last = e
            with _LOCK:
                _LEDGER["failures"] += 1
            if attempt >= max_retries or not _retryable(e):
                raise LLMError(f"{type(e).__name__}: {e}", status=_status_of(e)) from e
            time.sleep(delays[min(attempt, len(delays) - 1)])
    raise LLMError(f"{type(last).__name__}: {last}", status=_status_of(last))  # pragma: no cover


def preflight() -> tuple[bool, str]:
    """跑批前的一次真实调用。余额不足 / key 失效当场暴露，别让 2400 条任务白跑。"""
    try:
        chat([{"role": "user", "content": "ping"}], temperature=0.0, max_retries=0)
        return True, "ok"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
