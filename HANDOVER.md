# HANDOVER：项目一「Agent 检索策略消融与端到端评测」

> 本文档是**自包含交接书**。接手方无需读任何历史对话，照本文执行即可。
> 产出必须能被第三方复现，且能直接回答 6 个关键数字。

---

## 0. 一句话目标

在 HotpotQA（多跳问答）上，用**受控实验**量化「检索策略」对「Agent 端到端任务成功率 / 引用准确性 / 单次成本」的影响，输出一张带 95% 置信区间的消融表。

不是做产品，不是做 RAG 框架，**只做实验**。

---

## 1. Definition of Done（缺一条都算没做完）

- [ ] `eval/results/` 下有 **8 组 × 3 seed = 24 个** jsonl，每个 100 行
- [ ] `eval/results/summary.md` 有主表 + 相对基线（dense/重排off/改写off）的 delta 表，**每个 delta 都带 95% CI**
- [ ] `eval/results/manifest.json` 记录：模型名、单价、seed 列表、git sha、开始/结束时间、总成本
- [ ] `python eval/check.py` 全部断言通过
- [ ] 第 10 节「结果回填清单」的 6 个字段全部有真值
- [ ] `README.md` 写明：一条命令能重跑单组实验

---

## 2. 环境

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install datasets fastembed rank_bm25 openai numpy tqdm python-dotenv
```

**硬约束**

| 项 | 要求 | 原因 |
|---|---|---|
| LLM | DeepSeek（`base_url=https://api.deepseek.com`，`model=deepseek-chat`） | 国内直连，避免代理故障；成本约为 GPT 的 1/10 |
| Embedding | `fastembed` + `BAAI/bge-small-en-v1.5`（本地 ONNX，不用 GPU） | HotpotQA 是英文；避免装 torch |
| 向量检索 | **numpy 暴力内积**，不要装 faiss | 候选集只有几十条句子，faiss 是浪费 |
| Reranker | `fastembed.TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")` | 免 torch |
| 密钥 | 只从环境变量 `DEEPSEEK_API_KEY` 读，禁止写入源码 | — |

`.env.example` 里只放键名，不放真值。

---

## 3. 目录结构（不要自创别的布局）

```
rag-ablation/
├─ .env.example
├─ requirements.txt
├─ README.md
├─ data/hotpot_dev_100.jsonl
├─ app/
│  ├─ retrievers.py     # dense / bm25 / hybrid / rerank / query rewrite
│  ├─ llm.py            # 模型调用 + token 记账 + 重试
│  ├─ agent.py          # 多步循环
│  └─ metrics.py        # 六项指标 + bootstrap
└─ eval/
   ├─ build_dataset.py
   ├─ run.py            # 单组单 seed
   ├─ sweep.py          # 跑满 24 个组合
   ├─ analyze.py        # 出 summary.md
   ├─ check.py          # 断言
   └─ results/
```

---

## 4. 数据契约（不要改字段名，下游脚本依赖它）

`data/hotpot_dev_100.jsonl` 每行：

```json
{
  "qid": "5a8b57f25542995d1e6f1371",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "passages": [
    {"pid": 0, "title": "Scott Derrickson", "sent_id": 0, "text": "...", "is_gold": false},
    {"pid": 17, "title": "Ed Wood", "sent_id": 2, "text": "...", "is_gold": true}
  ],
  "gold_pids": [17, 43]
}
```

**构造规则**（`build_dataset.py`）

1. `load_dataset("hotpot_qa", "distractor", split="validation")`
2. `.shuffle(seed=42).select(range(100))` —— **必须固定 seed=42，禁止换**
3. 每个 `context` 段落按句子展开，一句 = 一条 passage，`pid` 为展开后全局自增序号
4. `is_gold=True` 当且仅当 `(title, sent_id)` 命中该样本的 `supporting_facts`
5. 断言：每条样本 `len(gold_pids) >= 2`，`len(passages) >= 30`

---

## 5. 模块规格

### 5.1 `retrievers.py`

```python
def build_index(passages: list[dict]) -> dict          # 建 BM25 + 向量矩阵，一次建好复用
def retrieve(index, query: str, k: int = 5, mode: str = "dense") -> list[dict]
    # mode ∈ {"dense","bm25","hybrid"}；返回 list[{pid,title,text,score}]
def rerank(query: str, cands: list[dict], k: int = 5) -> list[dict]
    # 对 top-20 重排取 top-k
def rewrite(query: str) -> list[str]
    # 返回 3 条子查询（含原 query）。失败时返回 [query]，不允许抛异常中断
```

- `hybrid` 用 RRF：`score = Σ 1/(60 + rank_i)`，k=**60**，不要改
- 每个检索器的候选池都是**同一条样本自己的 passages**（不许跨样本检索）
- 所有网络/模型异常必须 `try/except` 后降级为空结果，不允许让整条任务崩掉

### 5.2 `llm.py`

```python
def chat(messages, temperature=0.0, max_retries=3) -> dict
    # 返回 {"text": str, "prompt_tokens": int, "completion_tokens": int, "latency_ms": int}
```

- 指数退避重试（0.5s / 1s / 2s），仅对 429 / 5xx / 超时重试
- **每次调用都要累计 token 到全局账本**，供成本核算

### 5.3 `agent.py`

```python
def run_task(sample, cfg) -> dict
```

流程（`max_steps=8`）：

1. 若 `cfg.rewrite`：`queries = rewrite(question)`，否则 `queries=[question]`
2. 检索 → 可选 `rerank` → 取 top-5
3. 拼 prompt（[1]..[5] 编号 + 原文），要求模型输出 **严格 JSON**：
   `{"action": "answer" | "search", "answer": "...", "citations": [1,3], "query": "..."}`
4. `action=="search"` 且未超 `max_steps` → 用 `query` 再检索一轮，合并去重，回到 3
5. 终止：`action=="answer"` / 达 `max_steps` / 连续两轮无新证据
6. **JSON 解析失败**：重试 1 次，仍失败则记 `parse_error` 并用空答案结束（不要瞎猜）

返回：

```python
{"qid":..., "cfg":..., "seed":..., "answer": str, "citations": [int],
 "tool_calls": int, "latency_ms": int, "cost_cny": float,
 "prompt_tokens": int, "completion_tokens": int, "terminated_by": str}
```

**禁止**：把 `answer` / `gold_pids` / `supporting_facts` 的任何内容塞进 prompt。评测信息只能出现在 evaluate 阶段。

### 5.4 `metrics.py`

六项指标，口径**照抄，不许改**：

| 指标 | 分子 | 分母 |
|---|---|---|
| `f1` | token 级 F1（小写、去标点、按空格切分） | 1（逐条算完再平均） |
| `cit_p` | `citations` 中命中 `gold_pids` 的个数 | `len(citations)`（为空记 0） |
| `cit_r` | 同上 | `len(gold_pids)` |
| `success` | `f1 >= 0.5 and cit_p >= 0.5` → 1/0 | 1 |
| `p95_latency` | — | 成功任务的 `latency_ms` 第 95 百分位 |
| `cost_per_task` | 总 `cost_cny` | 成功任务数 |

```python
def bootstrap_ci(xs: list[float], n=1000, alpha=0.05, seed=0) -> tuple[float,float,float]
    # 返回 (mean, lo, hi)，用 percentile 法
```

- `citations` 里的数字是**段落序号 1..k**，要映射回 `pid` 再判断是否 gold
- 模型若返回超出范围的序号，判为错配，不计入分子但仍计入分母

---

## 6. 实验矩阵

固定：`model=deepseek-chat`、`temperature=0`、`top_k=5`、`max_steps=8`

```
retriever ∈ {dense, hybrid}
rerank    ∈ {off, on}
rewrite   ∈ {off, on}
seed      ∈ {42, 43, 44}
```

= 8 组 × 3 seed × 100 条 = **2400 次任务**

**先跑 1 组 1 seed 的 5 条**验证管线，再上全量。管线没过就全量 = 烧钱。

---

## 7. `summary.md` 输出格式（不要改表头）

```markdown
| config | success | cit_p | cit_r | f1 | p95_ms | cost_cny | Δsuccess vs base [95%CI] |
|---|---|---|---|---|---|---|---|
| dense / off / off (base) | 0.xx | ... | ... | ... | ... | ... | — |
| hybrid / on / off | 0.xx | ... | ... | ... | ... | ... | +x.x pt [+x.x, +x.x] |
```

必须附带：
- 失败四分类占比表（未召回 / 召回未读入 / 读入推理错 / 引用错配）
- 「gold 注入对照」：把 gold 段落直接塞进上下文重跑一次，报成功率 —— 用于把失败归因到检索侧
- 一段结论文字，明确写出**哪些增益落在 CI 内、不构成结论**

---

## 8. 禁止事项（违反即返工）

1. **禁止编造或微调数字**。所有数字必须来自 `eval/results/*.jsonl`。
2. **禁止用 LLM 当 judge**。金标只能是数据集自带的 `supporting_facts` 和 `answer`。
3. **禁止改指标口径**去让结果好看。口径写在 5.4，照抄。
4. **禁止只跑 1 个 seed 就下结论**。
5. **禁止把 `〔XX〕` 留空交付**。跑不出显著差异就如实写「无显著差异」。
6. 禁止引入 faiss / torch / 向量数据库 / 多 Agent。范围只有实验。

---

## 9. 成本与时间

| 项 | 预估 |
|---|---|
| 单次任务 LLM 调用 | 3–5 次 |
| 全量调用数 | ≈ 2400 × 4 ≈ 9600 次 |
| DeepSeek 成本 | **¥10–40**（按实际单价写入 manifest） |
| 全量串行时长 | 4–8 小时（可并发 4 线程，注意限流） |

时间表：D1 数据+基线 → D2 检索器全套 → D3 指标+单组验证 → D4 全量 → D5 bootstrap+出表。

**降级方案（只有 1 天时）**：100 条 × 4 组（dense/off/off、hybrid/off/off、hybrid/on/off、hybrid/on/on）× 1 seed，仍必须给 CI。

---

## 10. 结果回填清单（对外交付的 6 个数字）

跑完后必须回答这 6 个空位，**每个都附上来源 run_id**：

| # | 字段 | 取值来源 |
|---|---|---|
| 1 | 实验总成本 `〔¥XX〕` | `manifest.json → total_cost_cny` |
| 2 | 最优组合引用准确性 `〔XX〕%` | `summary.md` 中 cit_p 最高组 |
| 3 | 基线引用准确性 `〔XX〕%` | base 组 cit_p |
| 4 | 增益 `+〔XX〕pt` 与 `95% CI [〔XX〕, 〔XX〕]` | delta 表对应行 |
| 5 | 单次成本上升 `〔XX〕%` | 最优组 vs base 的 cost_cny 变化 |
| 6 | 失败四分类占比 | summary.md 归因表 |

**如果第 4 项的 CI 跨 0**：不要写「提升」，改写为「无显著差异，趋势方向为 X」，并换成另一条真实成立的结论。

---

## 11. 验收命令

```bash
python eval/build_dataset.py    # 产出 data/hotpot_dev_100.jsonl
python eval/run.py --mock --limit 3          # 不花钱验证管线
python eval/run.py --retriever dense --rerank off --rewrite off --seed 42
python eval/sweep.py --seeds 42,43,44
python eval/analyze.py
python eval/check.py            # 全绿才算完成
```

`--mock` 必须存在：用假响应（返回含 gold 句子前 3 个词的答案 + 第一个候选作为 citation）跑通全链路，**不调用任何 API**。

---

## 12. 交付时请回传

1. `eval/results/summary.md` 全文
2. `eval/results/manifest.json`
3. 第 10 节的 6 个值（附 run_id）
4. 一句结论：最优组合是什么、增益是否显著、瓶颈在哪一侧
