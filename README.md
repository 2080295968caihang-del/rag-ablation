# rag-ablation — Agent 检索策略消融与端到端评测

HotpotQA（distractor，固定 100 条）上量化**检索策略**对 Agent 端到端成功率 / 引用准确性 / 单次成本的影响。
产出 `eval/results/summary.md`（带 95% CI 的消融表）+ `manifest.json`（成本与可复现信息）。

## 一套命令重跑

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
cp .env.example .env            # 填 DEEPSEEK_API_KEY

python eval/build_dataset.py                        # 产出 data/hotpot_dev_100.jsonl（100 条 / 3883 段落）
python eval/run.py --mock --limit 3                 # 不花钱验证全链路
python eval/run.py --retriever dense --rerank off --rewrite off --seed 42   # 单组单 seed
python eval/sweep.py --seeds 42,43,44               # 8 组 × 3 seed = 2400 任务
python eval/analyze.py                              # 出 summary.md
python eval/check.py                                # 全绿才算完成
```

单组单 seed 一条命令（唯一必需参数就是这个）：

```bash
./.venv/Scripts/python.exe eval/run.py --retriever hybrid --rerank on --rewrite on --seed 43
# -> eval/results/hybrid_rr-on_rw-on_seed43.jsonl（100 行）
```

## 环境实测结论（不是猜的）

| 项 | 实测 | 处置 |
|---|---|---|
| `api.deepseek.com` | 模型列表 200，`deepseek-chat` 可用 | 直接用 |
| `deepseek-chat` 实际承接模型 | 响应 `model` 字段回 `deepseek-flash` | 官方价目表已只列 flash/pro，按 Flash off-peak cache-miss 价计（$0.15/$0.60 per 1M，汇率 7.1） |
| `deepseek-flash` 默认 | **thinking 模式**，`max_tokens` 小会返回空 content | 全程用 `deepseek-chat` 别名（实测非思考，JSON 输出稳定） |
| `huggingface.co` | 直连超时（10s 无响应） | `HF_ENDPOINT=https://hf-mirror.com`（已写进 `app/__init__.py` 默认值） |
| hf-mirror + Xet | `401 Unauthorized` at `cas-server.xethub.hf.co` | `HF_HUB_DISABLE_XET=1`（同上，默认值） |
| PyPI 直连 | 13 分钟只装了 pip | 装依赖走清华镜像 |
| `load_dataset("hotpot_qa", ...)` | datasets 5.1 报 `Repository id must be 'namespace/name'` | 直接拉 `hotpotqa/hotpot_qa` 的 `distractor/validation` parquet（同一份数据） |
| BAAI/bge-small-en-v1.5 + Xenova/ms-marco-MiniLM-L-6-v2 | 均下载成功、可推理 | 用 fastembed，无 torch |

## 目录

```
app/
  retrievers.py   dense / bm25 / hybrid(RRF k=60) / rerank / rewrite
  llm.py          deepseek-chat 调用 + token 账本 + 指数退避
  agent.py        多步循环（max_steps=8，严格 JSON，解析失败重试 1 次）
  metrics.py      f1 / cit_p / cit_r / success / p95 / cost + bootstrap + 失败四分类
eval/
  common.py       路径与命名（不是新分层，避免 4 个脚本各写一遍）
  build_dataset.py run.py sweep.py analyze.py check.py
```

## 口径与已知偏差（如实记，不美化）

1. **`passages >= 30` 的原始断言不成立**：distractor 分片里存在 1 句段落，seed=42 选出的 100 条中最少只有 6 条、16 条不足 30。
   断言降到 `>= 5`，**样本不动**（换 seed 会破坏可复现性）。最小样本池仍覆盖全部 gold。
2. **3 个 seed 高度相关（实测，不是假设）**：temperature=0 但服务端非严格确定性 —— base 组 3 seed 逐条
   答案相同 **91/100**、引用相同 **80/100**，约 9% 的轨迹会漂移。
   因此 CI 按 **qid 聚类**（n=100，先对 seed 取均值）算 bootstrap，不把 300 条当独立样本；interval 偏保守。
3. **成本单价是「off-peak cache-miss」口径**：DeepSeek 峰时（UTC 01:00-04:00、06:00-10:00 工作日）价格翻倍，
   实际账单可能比 `manifest.json` 高不到 2 倍；`manifest.json → prices` 里写死了所用单价。
4. **重排模型不可用时静默降级为「原序截断」**，检索不可用时降级为空结果（HANDOVER 5.1 要求不中断），
   `run.py` 启动会打印 `[models]` 状态，`manifest` 里不含该项，验收前请确认 `rerank_ok: true`。
5. **API 级故障已改为 fail-fast（2026-11-02 事故后）**：余额不足（402）曾被 `except Exception` 当成
   `parse_error` 吞掉，导致 5 个结果文件里 2400+ 条是静默垃圾。现在 `llm.LLMError` 上抛、
   `run.py` 开跑前做一次 `preflight()` 真实调用、失败自动删除半份文件并以退出码 3 中止、
   `check.py` 断言结果中不得出现 `parse_error` / `api_error`。**跑批前先看余额**。

## 成本

单任务 1–4 次 LLM 调用。全量 2400 任务 + 100 条 gold 对照实测 **¥9.22**（`manifest.json → total_cost_cny`），
耗时 28 分钟（workers=4）。单组单 seed 100 条约 ¥0.37–0.47。

## 结论（一句话）

8 组之间 success / cit_p / cit_r 的 delta **全部 CI 跨 0**，检索策略对端到端成功率的提升不成立；
唯一 CI 不跨 0 的是 `hybrid / off / off` 相对 base 的**成本 -24.2%**（[-40.0%, -10.7%]）与 p95 延迟 -1.22 s。
gold 注入对照 0.880 vs base 0.707，说明约 17.3 pt 丢在检索侧、其余 12% 丢在读入后的推理与引用。
