# rag-ablation — Agent 检索策略消融与端到端评测

HotpotQA（distractor，固定 600 条）上量化**检索策略**对 Agent 端到端成功率 / 引用准确性 / 单次成本的影响。
产出 `eval/results_600/summary.md`（带 95% CI 的消融表）+ `manifest.json`（成本与可复现信息）。

## 一套命令重跑

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
cp .env.example .env            # 填 DEEPSEEK_API_KEY（35 位，sk- 开头，不加引号）

./.venv/Scripts/python.exe eval/build_dataset.py --n 600   # 产出 data/hotpot_dev_600.jsonl（600 条 / 24.5k 段落）
./.venv/Scripts/python.exe eval/run.py --mock --limit 3    # 不花钱验证全链路
./.venv/Scripts/python.exe eval/sweep.py --seeds 42,43 --workers 4   # 8 组 × 2 seed × 600 条 = 9600 任务
./.venv/Scripts/python.exe eval/analyze.py                 # 出 summary.md
./.venv/Scripts/python.exe eval/check.py                   # 全绿（31 项）才算完成
```

单组单 seed 一条命令：

```bash
./.venv/Scripts/python.exe eval/run.py --retriever hybrid --rerank on --rewrite on --seed 43
# -> eval/results_600/hybrid_rr-on_rw-on_seed43.jsonl（600 行）
```

`eval/common.py` 里的 `DATA` / `RESULTS` 默认已指向 600 条那套；所有脚本都支持 `--data` / `--results` 覆盖。

## 暂停 / 续跑（关机前用）

**暂停**（任意时刻，不丢已跑数据）：

```bash
touch STOP          # 在项目根目录建 STOP 文件
```

- `run.py` 在**每批任务边界**检查 `STOP`：本批（workers 条）跑完、结果落盘后才停，退出码 4。
- `sweep.py` 看到退出码 4 立即停止整个 sweep，后续组合不启动。
- 已写的结果**不删、不覆盖**；半份文件保留。

**续跑**（删掉 STOP，原命令照跑）：

```bash
rm STOP && ./.venv/Scripts/python.exe eval/sweep.py --seeds 42,43 --workers 4
```

- 已满 600 行的文件直接 `[skip]`。
- 半份文件 `[resume] 已有 N 条，跳过已完成 qid`，只补缺的。
- 坏行（硬关机砍断的残行）自动剔除并备份为 `*.jsonl.bad`，不会让整份结果作废。

强制从零重跑：`--force`（sweep 会先删文件，并给 run.py 传 `--no-resume`）。

## 环境实测结论（不是猜的）

| 项 | 实测 | 处置 |
|---|---|---|
| `api.deepseek.com` | 模型列表 200，`deepseek-chat` 可用 | 直接用 |
| `deepseek-chat` 实际承接模型 | 响应 `model` 字段回 `deepseek-flash` | 官方价目表已只列 flash/pro，按 Flash off-peak cache-miss 价计（$0.15/$0.60 per 1M，汇率 7.1） |
| `deepseek-flash` 默认 | **thinking 模式**，`max_tokens` 小会返回空 content | 全程用 `deepseek-chat` 别名（实测非思考，JSON 输出稳定） |
| `huggingface.co` | 直连超时（10s 无响应） | `HF_ENDPOINT=https://hf-mirror.com`；**但运行期并不需要**，模型已随项目放在 `models/` |
| hf-mirror + Xet | `401 Unauthorized` at `cas-server.xethub.hf.co` | `HF_HUB_DISABLE_XET=1` |
| PyPI 直连 | 13 分钟只装了 pip | 装依赖走清华镜像 |
| `load_dataset("hotpot_qa", ...)` | datasets 5.1 报 `Repository id must be 'namespace/name'` | 直接拉 `hotpotqa/hotpot_qa` 的 `distractor/validation` parquet（同一份数据） |
| BAAI/bge-small-en-v1.5 + Xenova/ms-marco-MiniLM-L-6-v2 | 均下载成功、可推理；断网（`HF_HUB_OFFLINE=1`）亦可 | 缓存固化到 `models/`（152MB，`FASTEMBED_CACHE_PATH` 可覆盖） |

## 目录

```
app/
  retrievers.py   dense / bm25 / hybrid(RRF k=60) / rerank / rewrite
  llm.py          deepseek-chat 调用 + 线程局部 token 计量 + 指数退避（5xx 全重试）
  agent.py        多步循环（max_steps=8，严格 JSON，解析失败重试 1 次）
  metrics.py      f1 / cit_p / cit_r / success / p95 / cost + bootstrap + 失败四分类
eval/
  common.py       路径与命名（不是新分层，避免 4 个脚本各写一遍）
  build_dataset.py run.py sweep.py analyze.py check.py
  results_600/    最终结果（16 个消融文件 + gold 注入 + manifest + summary）
  _archive/       已作废的历史结果，见下文「事故记录」
models/           嵌入 + 重排模型缓存（gitignore）
```

## 口径与已知偏差（如实记，不美化）

1. **`passages >= 30` 的原始断言不成立**：distractor 分片里存在 1 句段落，seed=42 选出的样本中最少只有 6 条。
   断言降到 `>= 5`，**样本不动**（换 seed 会破坏可复现性）。最小样本池仍覆盖全部 gold。
2. **2 个 seed 高度相关（实测，不是假设）**：temperature=0 但服务端非严格确定性 —— base 组 2 seed 逐条
   答案相同 **553/600**、引用相同 **541/600**，约 8% 的轨迹会漂移。
   因此 CI 按 **qid 聚类**（n=600，先对 seed 取均值）算 1000 次配对 bootstrap，不把 1200 条当独立观测；区间偏保守。
3. **只跑 2 个 seed 而非 3 个**：余额限制（3 seed 需约 ¥14，当时可用 ¥49.97 但要留余量）。统计功效主要来自
   n=600 的 qid 聚类，不是因为 seed 数量；seed 只影响轨迹扰动，不影响聚类单元。
4. **成本单价是「off-peak cache-miss」口径**：DeepSeek 峰时（UTC 01:00-04:00、06:00-10:00 工作日）价格翻倍，
   实际账单可能比 `manifest.json` 高不到 2 倍；`manifest.json → prices` 里写死了所用单价。
   本次实测：跑前余额 ¥38.84、跑完（含 gold）¥29.13，账单扣减 **¥9.71** vs manifest **¥9.56**，差 1.6%（预检探测与重试），量级一致。
5. **模型不可用已改为 fail-fast**：嵌入模型加载失败、或 `rerank=on` 但重排模型加载失败，`run.py` 直接
   退出码 3，不再静默降级成「空结果 / 原序截断」。`[models]` 行会打印 `embedding_ok` / `rerank_ok` / `cache_dir`。
6. **API 级故障已改为 fail-fast（2026-11-02 事故后）**：余额不足（402）曾被 `except Exception` 当成
   `parse_error` 吞掉，导致 5 个结果文件里 2400+ 条是静默垃圾。现在 `llm.LLMError` 上抛、
   `run.py` 开跑前做一次 `preflight()` 真实调用、失败时保留已写结果并以退出码 3 中止、
   `check.py` 断言结果中不得出现 `parse_error` / `api_error`。**跑批前先看余额**。
7. **5xx 一律重试**：520/521/522 等 Cloudflare 系列属于上游瞬时故障。此前白名单只有 500/502/503/504，
   一次 520 就把整批 sweep 打成中止（已发生一次，351 条时中断）。现在是 `429 / 408 / 500–599` 全重试，
   退避 1/2/4/8/16 s，最多 5 次。

## 事故记录：成本计量 bug（已修复，数据已重跑）

- **现象**：`agent.py` 用全局 ledger 差分算单任务用量（`before = snapshot()` → 跑 → 差值）。workers=4 时
  4 个线程同时在花 token，差值里混进别人的账，**成本被放大 2.19–2.62 倍**，且改写组（调用次数多）
  放大得更厉害（2.6x vs 2.2x）。
- **后果**：基于旧数据得出的「混合检索降本 24%（100 条）/ 48%（600 条）」**全部作废**。这是假象，
  不是结论。成功率 / 引用类指标不受影响（不依赖 token 计数）。
- **修复**：`llm.chat` 本就返回准确的 per-call usage，改为**线程局部累加器**（`task_reset` / `task_usage`），
  一个线程只跑一条任务。验证：20 条任务（workers=4）记录求和 `15086+1613` 与 ledger 完全一致。
- **处置**：旧结果移到 `eval/_archive/results_600_costbug/` 与 `eval/_archive/results_100_costbug/`，
  最终结论只基于 `eval/results_600/`。出表前核对「逐条 cost 求和 == manifest == 账单扣减」三者一致。

## 成本

全量 9600 任务 + 600 条 gold 对照实测 **¥9.56**（`manifest.json → total_cost_cny`，其中消融 ¥9.32、
gold ¥0.24），耗时 179 分钟（workers=4）。单任务 ¥0.00082–0.00113，视配置而定。

## 结论（一句话）

检索策略对端到端成功率的增益**不显著**（最好的 `hybrid / off / off` 相对 base 为 +1.7 pt，95% CI [-1.3, +4.7]），
但**显著降本**（-13.1%，CI [-21.3%, -5.3%]）；**查询改写是纯负向项**（success -3.1 pt、cit_p -1.7 pt、成本 +29.3%，
四项 CI 均不跨 0）；gold 注入对照 0.845 vs base 0.674，说明 17.1 pt 丢在检索侧、其余 15.5% 丢在读入后的推理与引用。
