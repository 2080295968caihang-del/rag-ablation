# rag-ablation — Agent 检索策略消融与端到端评测

HotpotQA（distractor）上量化**检索策略**对 Agent 端到端成功率 / 引用准确性 / 单次成本的影响。

> 数据构造、每个参数的取值理由、技术栈选型与被排除的替代方案：见 [`DESIGN.md`](DESIGN.md)。
> 未验证项、实验设计与「结果出来后对外结论怎么改」：见 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)。
> 三种长度的结论摘要（一句话 / 精简 / 完整）与设计决策问答：见 [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md)。

六阶段，共 **30,600 次 Agent 任务、总成本 ¥34.11**（已去重：阶段 2 的前 600 题是阶段 1 的复制，只计一次）：

| 阶段 | 问题 | 规模 | 产物 |
|---|---|---|---|
| 1 筛选 | 8 组「检索 × 重排 × 改写」有差异吗 | 600 题 × 2 seed × 8 组 + gold 对照（¥9.56） | `eval/results_600/summary.md` |
| 2 确认 | 最好的那组是真效应还是选组偏差 | 扩到 1500 题，留出 900 题未参与选组（新增 3,600 条 / ¥3.48） | `eval/results_confirm_1500/confirm.md` |
| 3 干预 | 瓶颈到底在哪 | top_k = 5/10/20/30/40/50/全部候选，同组同题配对（7,200 条 / ¥11.27） | 同上 §3–§5 |
| 4 多步价值 | 多步循环值不值 | max_steps 1 vs 8，同检索同 top_k（¥0.78） | 同上 §6 |
| 5 留出复现 | k 的效应是不是只在选组那 600 题上 | 900 道未参与选择的题上重跑 k=20 + 同批 k=5 基线（3,600 条 / ¥4.05） | 同上 §7 |
| 6 加固 | 结论对配置/模型/seed 是否稳健 | dense 交叉 1,200 + 第 3 个 seed 1,200 + 换嵌入 1,200 + 换重排 1,200（¥4.97） | `eval/results_cross/cross_config.md`、`eval/results_topk/extra_arms.md` |

```bash
./.venv/Scripts/python.exe eval/check.py          # 91 项断言，全绿才算完成
```

## 数据来源与许可

- **数据集**：HotpotQA dev 分片的 **distractor** 设定（上游 <https://hotpotqa.github.io/>，许可 CC BY-SA 4.0）。
  本仓库不附带原始全文，只用 `eval/build_dataset.py` 从 dev 分片抽样构造四份固定子集
  （100 / 600 / 1,500 / holdout 900）；后三份为严格超集，抽样规则与 `seed` 见各 `manifest.json`。
  原始问题与 supporting facts **未做修改**，只做抽样与打散，因此本仓库的指标不能直接与 HotpotQA 官方榜单比较。
- **嵌入 / 重排模型**：`BAAI/bge-small-en-v1.5`、`Xenova/ms-marco-MiniLM-L-6-v2`（fastembed 默认模型，
  许可见各自 HF 页面）。**不随仓库分发**：首次运行需联网下载（默认两份模型合计约 152MB），缓存到 `models/`
  （已 gitignore，不入库）。因此**新克隆的仓库必须至少联网一次**才能跑检索与重排；下载完成后可断网运行
  （`HF_HUB_OFFLINE=1` 实测通过）。复现「换模型」两档会额外下载 `bge-base` / `MiniLM-L-12`（约 338MB）。
- **代码许可**：MIT，见 [`LICENSE`](LICENSE)。

## 一套命令重跑

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
cp .env.example .env            # 填 DEEPSEEK_API_KEY（35 位，sk- 开头，不加引号）

./.venv/Scripts/python.exe eval/build_dataset.py --n 600   # 产出 data/hotpot_dev_600.jsonl（600 条 / 24.5k 段落）
./.venv/Scripts/python.exe eval/run.py --mock --limit 3    # 不花钱验证全链路
./.venv/Scripts/python.exe eval/sweep.py --seeds 42,43 --workers 4   # 8 组 × 2 seed × 600 条 = 9600 任务
./.venv/Scripts/python.exe eval/analyze.py                 # 出 summary.md
./.venv/Scripts/python.exe eval/check.py                   # 全绿（91 项）才算完成
```

单组单 seed 一条命令：

```bash
./.venv/Scripts/python.exe eval/run.py --retriever hybrid --rerank on --rewrite on --seed 43
# -> eval/results_600/hybrid_rr-on_rw-on_seed43.jsonl（600 行）
```

`eval/common.py` 里的 `DATA` / `RESULTS` 默认已指向 600 条那套；所有脚本都支持 `--data` / `--results` 覆盖。

## 第二阶段（扩样确认）与第三阶段（top_k 干预）

```bash
# 1500 题数据集（前 600 条与 600 集逐条一致，严格超集）
./.venv/Scripts/python.exe eval/build_dataset.py --n 1500 --out data/hotpot_dev_1500.jsonl

# 只跑「基线 + 最好那组」，1500 题 × 2 seed
for sd in 42 43; do
  ./.venv/Scripts/python.exe eval/run.py --retriever dense  --rerank off --rewrite off --seed $sd \
    --data data/hotpot_dev_1500.jsonl --out eval/results_confirm_1500/dense_rr-off_rw-off_seed${sd}.jsonl
  ./.venv/Scripts/python.exe eval/run.py --retriever hybrid --rerank on  --rewrite off --seed $sd \
    --data data/hotpot_dev_1500.jsonl --out eval/results_confirm_1500/hybrid_rr-on_rw-off_seed${sd}.jsonl
done

# top_k 干预：同组同题，只改槽位数
for k in 10 20 40; do
  for sd in 42 43; do
    ./.venv/Scripts/python.exe eval/run.py --retriever hybrid --rerank on --rewrite off \
      --top-k $k --seed $sd --out eval/results_topk/hybrid_rr-on_rw-off_top${k}_seed${sd}.jsonl
  done
done

# 阶段 5：在 900 道未参与选择的题上复现 k=20。k=5 基线要**同时重跑一遍**，
# 否则就是跨批次比较（见 confirm.md §7.2 的噪声表）；这份重跑顺带量出重复运行误差。
tail -n 900 data/hotpot_dev_1500.jsonl > data/hotpot_dev_holdout900.jsonl
for sd in 42 43; do
  ./.venv/Scripts/python.exe eval/run.py --retriever hybrid --rerank on --rewrite off \
    --top-k 20 --seed $sd --data data/hotpot_dev_holdout900.jsonl \
    --out eval/results_holdout/hybrid_rr-on_rw-off_top20_seed${sd}.jsonl
  ./.venv/Scripts/python.exe eval/run.py --retriever hybrid --rerank on --rewrite off \
    --top-k 5 --seed $sd --data data/hotpot_dev_holdout900.jsonl \
    --out eval/results_holdout_fresh/hybrid_rr-on_rw-off_top5_seed${sd}.jsonl
done

# 出 confirm.md（自动发现 results_topk 下所有 k，逐档配对检验 + 饱和检查 + 留出复现）
./.venv/Scripts/python.exe eval/confirm.py --topk-dir eval/results_topk
```

`confirm.py` 的 bootstrap 按「指标 + qid 列表」**确定性播种**，所以同一份数据的 CI 每次重算都一样；
不播种会让 CI 随"在它之前算了多少个量"漂移，加一节分析就会让前面的区间变一点。

`confirm.py` 出表前会校验每个 seed 文件的**行数**，不齐的档位直接标"跳过"而不是拿半份数据算均值
（本轮的 top_k=20/seed43 就曾只有 540/600，补齐后才进表）。


## 暂停 / 续跑

**暂停**：

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
- 坏行自动剔除并备份为 `*.jsonl.bad`，不会让整份结果作废。

强制从零重跑：`--force`（sweep 会先删文件，并给 run.py 传 `--no-resume`）。

## 环境实测结论

| 项 | 实测 | 处置 |
|---|---|---|
| `api.deepseek.com` | 模型列表 200，`deepseek-chat` 可用 | 直接用 |
| `deepseek-chat` 实际承接模型 | 响应 `model` 字段回 `deepseek-flash` | 官方价目表已只列 flash/pro，按 Flash off-peak cache-miss 价计（$0.15/$0.60 per 1M，汇率 7.1） |
| `deepseek-flash` 默认 | **thinking 模式**，`max_tokens` 小会返回空 content | 全程用 `deepseek-chat` 别名（实测非思考，JSON 输出稳定） |
| `huggingface.co` | 直连超时（10s 无响应） | `HF_ENDPOINT=https://hf-mirror.com`；仅**首次下载**需要，之后运行期不需要（缓存在本地 `models/`） |
| hf-mirror + Xet | `401 Unauthorized` at `cas-server.xethub.hf.co` | `HF_HUB_DISABLE_XET=1` |
| PyPI 直连 | 13 分钟只装了 pip | 装依赖走清华镜像 |
| `load_dataset("hotpot_qa", ...)` | datasets 5.1 报 `Repository id must be 'namespace/name'` | 直接拉 `hotpotqa/hotpot_qa` 的 `distractor/validation` parquet（同一份数据） |
| BAAI/bge-small-en-v1.5 + Xenova/ms-marco-MiniLM-L-6-v2 | 下载成功、可推理；**下载完成后**断网（`HF_HUB_OFFLINE=1`）亦可 | 缓存在本地 `models/`（152MB，不入库 → 新克隆需先联网一次；`FASTEMBED_CACHE_PATH` 可覆盖） |

## 目录

```
app/
  retrievers.py   dense / bm25 / hybrid(RRF k=60) / rerank / rewrite
  llm.py          deepseek-chat 调用 + 线程局部 token 计量 + 指数退避（5xx 全重试）
  agent.py        多步循环（max_steps=8，严格 JSON，解析失败重试 1 次）
  metrics.py      f1 / cit_p / cit_r / success / p95 / cost + bootstrap + 失败四分类
eval/
  common.py       路径与命名（不是新分层，避免 4 个脚本各写一遍）
  build_dataset.py run.py sweep.py analyze.py check.py confirm.py
  results_600/          阶段 1：16 个消融文件 + gold 注入 + manifest + summary
  results_confirm_1500/ 阶段 2/3：1500 题两组的 4 个文件 + confirm.md
  results_topk/         阶段 3：top_k 10/20/40 × 2 seed
  results_steps/        阶段 4：max_steps=1 单步对照
  results_holdout/      阶段 5：留出 900 题上复现 k=20
  results_holdout_fresh/ 阶段 5：留出 900 题的同批 k=5 基线（估重跑噪声）
  _archive/             已作废的历史结果，见下文「事故记录」
models/           嵌入 + 重排模型缓存（gitignore）
```

## 口径与已知偏差

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
8. **top_k 干预的跨配置验证只补了一组**：除 `hybrid+rerank` 外，本次只在 `dense/off/off` 上重跑 k=20
   （600 题 × 2 seed，¥1.6），得到 success **+11.83 pt** [+8.91, +14.83]、覆盖率 +21.62 pt、成本 +45.0%——
   **比 hybrid 的 +10.50 还略大**，说明「槽位数是杠杆」不依赖重排。其余 6 组仍未交叉验证，
   所以能说的是「在两种检索配置上均复现」，不能说的是「8 组都以 k=20 为最优」。
   **留出集已补（见 §7）**：900 道未参与选择的题上、**与 k=20 同一天跑的同批 k=5 基线**对比，
   success **+8.50 pt** [+6.11, +10.89]、覆盖率 **+22.52 pt**，方向和显著性都复现，
   但**效应量比选组那 600 题的 +10.50 pt 小 2 pt（约 19%）**——
   对外表述应以留出集的 +8.5 pt 为准，600 题的 +10.5 是上界。
9. **`k=40` 那一档是无效对照（已实测确认，不是推测）**：`app/agent.py:116` 写的是
   `retrieve(..., k=20 if rerank else top_k)`，开重排时**每一步**只取 20 条候选，`top_k>20` 只能靠跨步累积
   增加上下文（`agent.py:180` 会用模型给的新 query 再检索，候选池跨步合并）。实测 k=20 平均 19.9 条、
   k=40 平均 20.1 条，仅 3.1% 的任务超过 20 条、最多 32 条——差异太小，**不能当饱和证据**。
   旧代码下「k=20 之后饱和」这句话不成立，因为 k>20 的档根本没拿到更多候选。
   该常数**已提成参数**（`--cand-k`，强制 `cand_k >= top_k`）并补测：k=30 边际 Δsucc **+0.00 pt** [−1.42,+1.34]、
   覆盖率仍涨 4.4 pt，而 k=30 成本 +93.9%（k=50 / 全量另有 +1.08 / +0.25 pt，均不显著）。
   因此结论只能写成「**本实验中 k=20→30 未观察到显著成功率提升**」——这是本管线候选池配置 + 固定模型下的观测，
   不是「k>20 普遍无用」的普适结论。
10. **k=20 的 success 0.789 仍低于全 gold 时的 0.825–0.860**，因为 93.6% 是平均覆盖率，仍有一部分题缺 gold。
    差距（约 3.5–7 pt）就是剩余检索损失 + 引用错配，不是"槽位还不够"。
11. **多步对照有一个设计副作用**：`max_steps=1` 时若模型仍要求追加检索，循环会退出且答案为空——
    已加「步数耗尽强制收口一次」（`terminated_by=forced_answer`），否则单步组会被系统性压低成假负效应。
    全量 23,400 条记录里只有 36 条（0.15%）命中 `max_steps`；单步组 1,200 条里有 **263 条（21.9%）** 靠这个兜底
    才拿到答案 —— 与多步组「21.6% 的任务需要 >1 步」两个独立口径吻合。旧数据未重跑。
12. **`cit_p` 是精度，不含召回**：success 只卡 `cit_p >= 0.5` 与 `f1 >= 0.5`，**不卡 `cit_r`**。
    实测平均引用条数 k=5 为 **1.93**、k=20 为 **2.24**（每题 gold 均值 2.45）；只引 1 条的任务占比
    由 **21.9% 降到 5.8%**（这些任务 success 54.4% / 59.4%），`cit_r` 全程约 0.64。
    所以「引用准确率 0.86」不等于「引用齐全」——这是本指标的已知弱点；不把 `cit_r` 并进 success
    是为了避免与 f1 重复计分（f1 已要求答案对，引用召回再加一道会双重惩罚）。

## 事故记录：成本计量 bug

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

六阶段实测 **¥34.11 / 30,600 条**（逐条 `cost_cny` 求和且**去重**；阶段 1 与账单核对过，差 1.6%）：

| 阶段 | 任务数 | 成本 | 备注 |
|---|---|---|---|
| 8 组消融 + gold 对照 | 10,200 | ¥9.56 | 9600 + 600，179 分钟 |
| 扩样 1500（2 组 × 2 seed） | 6,000 | ¥5.62 | 其中 **2,400 条是阶段 1 的复制**（前 600 题同配置同 seed，逐字节相同，不重复计费），实际新增 3,600 条 / ¥3.48 |
| top_k 10/20/30/40/50/全量（6 档 × 2 seed × 600 题） | 7,200 | ¥11.27 | top_k=5 基线直接取扩样那批的同题结果；其中 30/50/全量三档 ¥6.89 是本轮新跑 |
| max_steps=1 单步对照（2 seed × 600 题） | 1,200 | ¥0.78 | |
| 留出 900 题复现 k=20（2 seed × 900 题） | 1,800 | ¥2.40 | 上下文 19.9 条，与阶段 3 的 k=20 一致 |
| 留出 900 题的同批 k=5 基线（2 seed × 900 题） | 1,800 | ¥1.65 | 只为让 §7 的自洽比较同批成立，同时量出重跑噪声 |
| 阶段 6 加固（dense 交叉 / 第 3 seed / 换嵌入 / 换重排） | 4,800 | ¥4.97 | 每项 1,200 条；换模型两组还含模型下载（¥0） |

单任务 ¥0.00086（k=5）→ ¥0.00131（k=20）→ ¥0.00207（全量候选）。
账户实际扣费与逐条估算的差额：本轮 ¥20.70 → ¥12.24（扣 ¥8.46），逐条估算新增 ¥10.87，
**估算比账单高 28%**（估算按 cache-miss 价，实际有命中）。对外报成本用账单口径或直接说区间。


## 结论

1. **检索策略本身不是瓶颈。** 8 组配置的 success 差异都在噪声内（最好 +1.7 pt，CI [-1.3, +4.7]）。根因测得清楚：
   各组的 gold 覆盖率都在 **70–74%**，全 gold 进上下文时 success **0.818–0.860**，缺 gold 时只有 **0.500–0.576**，
   `0.486×0.825 + 0.514×0.532 = 0.674` 与实测 0.674 吻合。策略换来换去，没换到覆盖率。
2. **改写是纯负向项，砍掉。** success **-3.1 pt** [-5.6, -0.5]、cit_p **-1.7 pt** [-3.3, -0.1]、成本 **+29.3%**、
   p95 **+827 ms**，四项同向且均不跨 0。这是唯一被数据直接支持的组合决策。
3. **真正的杠杆是上下文槽位数，不是检索算法。** top_k 5→20：gold 覆盖率 **71.3%→93.6%**（+22.3 pt）、
   success **0.684→0.789**（**+10.5 pt** [+7.7, +13.4]）、成本 +52.5%。机制测得清：600 题的失败记录 379→253，
   `召回未读入` 从 68 条归零；k=5 时归入该型的 **34 道题（两 seed 去重后）有 23 道**在 k=20 转为成功（67.6%）。
   **去留出集复现（§7）**：900 道未参与选择的题上 success **+8.50 pt** [+6.11, +10.89]、覆盖率 **+22.52 pt**、
   cit_p +4.73 pt——方向与量级都复现，但效应量比选组那 600 题小 **2 pt（约 19%）**，所以对外报数用 +8.5。
   **饱和点已测到（k=20）**：把候选池提成 `--cand-k` 后重跑 k=30 / k=50 / 全部候选（每档 600 题 × 2 seed），
   success 变成 0.789 → 0.789 → 0.800 → 0.802，对应边际 Δsucc **+0.00** [−1.42,+1.34]、**+1.08** [−0.17,+2.42]、
   **+0.25** [−0.42,+0.84]（后两档都不显著），同期 gold 覆盖率从 93.6% 一路涨到 **100%**、成本却从 +52.5% 涨到 **+141.6%**。
   即 **k=20 之后再给槽位只买覆盖率、不买成功率**。
   **顺带得到一个天花板**：把该题全部候选（平均 40.9 条）塞进上下文、覆盖率 100% 时，success 也只有 **0.802**——
   在本数据、`deepseek-chat` 与这套提示下，检索链路做到完美也只到 0.80，剩下约 20% 是「gold 已在上下文里仍答错/引错」，
   不是加槽位能修的。这是**本管线的观测上限**（覆盖率 100% 时实测），不是 HotpotQA 或该模型的理论上限。
   **阈值不是拍脑袋**：success 的 0.5 做过敏感性（`eval/sensitivity.py`，离线 ¥0），T 从 0.30 扫到 0.70，
   Δ 在 +4.9~+11.8 pt 之间且五档 CI 全不跨 0——结论不依赖这个具体取值。

补充：`hybrid / off / off` 显著降本（-13.1%，CI [-21.3, -5.3]）；`hybrid+rerank` 的增益要扩样后才站得住——
**全集 1,500 题** cit_p **+2.06 pt** [+0.69, +3.42]、success **+2.10 pt** [+0.33, +3.87]；其中**未参与选组的
900 题**上 cit_p **+2.13 pt** [+0.47, +3.94]、success **+2.83 pt** [+0.78, +5.11]——即该组在 n=600 上
「趋势 +2 pt 但 CI 跨 0」不是假效应，只是 n=600 功效不足。

4. **多步循环值 +6.9 pt。** 把上限压到 `max_steps=1`（只允许检索一次就当答）：success **0.684→0.615**
   （**−6.9 pt** [-9.1, -4.7]）、gold 覆盖率 71.3%→66.0%、成本 −24.2%。即多步净价值 +6.9 pt / +24% 成本
   （每任务多 ¥0.0002）。注意同组有 **78.4% 的任务本来 1 步就收口**（平均 1.50 步），所以这 6.9 pt 是
   由 21.6% 的任务撑起来的——改写组平均 3.47 步，这也是它成本 +29.3% 的机制之一，不只是 prompt 变长。

5. **结论对「换配置 / 换模型 / 换 seed」都不敏感。** 三项加固各 1,200 条：
   - 换检索配置：`dense/off/off` 上 k 5→20 仍有 **+11.83 pt** [+8.91, +14.83]，比 hybrid 的 +10.50 还略大
     → 槽位数这条杠杆**不依赖重排**；
   - 换模型：`bge-small→bge-base`（384→768 维）覆盖率 **+0.03 pt**、success **+0.08 pt**；
     `MiniLM-L-6→L-12` 覆盖率 +0.34 pt、success −0.67 pt，CI 全跨 0 → **70–74% 的覆盖率上限是任务/标注特性，不是嵌入模型的锅**
     （旁证：`eval/model_sanity.py` 证明 L-6 与 L-12 在同一批候选上 top-5 有 3/5 题不同，模型确实换了）；
   - 加第 3 个 seed：Δ **+10.44 pt** [+7.61, +13.22]（原 +10.50），CI 宽度只缩小 **2.4%**
     → 区间宽度由 600 题决定，不由 seed 数决定（按 qid 聚类的必然结果）。

## 结论速查

**一句话**：在 HotpotQA distractor 上做 Agent 检索策略消融（30,600 次任务 / ¥34.11），用 qid 聚类 bootstrap
+ 未参与选组的留出集确认：混合检索+重排的增益在 n=600 下 CI 跨 0、扩样到 1,500 题后才站住，但只有 **+2.1 pt**
（success，全集；未参与选组的 900 题为 +2.8 pt），而仅仅把上下文槽位数从 5 提到 20 就有 **+10.5 pt**（gold 覆盖率 71%→94%）、
在 900 道未参与选择的题上复现为 **+8.5 pt**、在 dense 配置上复现为 **+11.8 pt**，且 **k=20 就是拐点**
（k=30/50/全量都不显著、成本翻倍）；据此砍掉查询改写（-3.1 pt / +29% 成本），并把优化重心从"换检索算法"移到"给够上下文"。

**背景 / 目标 / 做法 / 结果**：

- **背景**：RAG 系统里"检索策略该怎么选"通常靠直觉。要回答的是：在固定模型与预算下，换检索器 / 加重排 / 加改写
  到底能不能提升端到端成功率，提升多少，代价多少。
- **目标**：必须给出带置信区间的结论，且要能区分「真效应」与「选组偏差」「计量 bug」。
- **做法**：8 组配置 × 2 seed × 600 题的消融；发现 8 组无差异后不硬报趋势，而是做 gold 注入对照把失败归因成
  「未召回 / 召回未读入 / 读入推理错 / 引用错配」四类，测出检索侧天花板是覆盖率；再扩样到 1500 题并保留
  900 题完全不参与选组；最后用 top_k 干预验证因果。中途发现 token 计量在并发下串账（成本虚高 2.2–2.6x），
  改为线程局部累加器并重跑全部数据。
- **结果**：改写 -3.1 pt（CI 不跨 0）；top_k 5→20 成功率 +10.5 pt（CI [+7.7, +13.4]），并在 900 道未参与选择的题上
  复现为 +8.5 pt（CI [+6.1, +10.9]）、在 dense 配置上复现为 +11.8 pt（CI [+8.9, +14.8]）；扩样确认 hybrid+rerank
  **+2.1 pt**（全集 1,500，CI [+0.33, +3.87]）/ **+2.83 pt**（留出 900，CI [+0.78, +5.11]）。
  另测到饱和点 k=20（k=30/50/全量边际 Δsucc 均不显著、成本 +93.9%/+134.7%/+141.6%），以及「覆盖率 100% 时
  success 也只有 0.802」这一**本管线内的观测上限**，把剩余误差明确归给推理与引用而非检索。
  全部结论由 91 项断言脚本（跨文档数字与报告现算值比对）+ manifest（sha / 单价 / 逐条成本）背书。

**部分工程细节**：

- CI 按 **qid 聚类**算，不把 2 个 seed 的 1200 条当独立观测（实测两 seed 逐条答案相同 553/600，约 8% 漂移）。
- 并发下用**线程局部累加器**计量，核对「逐条 cost 求和 == manifest == 账单扣减」三者一致（差 1.6%）。
- **量过重跑噪声**：同配置、同 seed、隔 4 天重跑 900 题，success 差 **0.00 pt**（CI [-0.94, +0.95]）、
  gold 覆盖率差 −0.17 pt、成本差 −2.2%，但 cit_p 有 **+0.74 pt** [+0.01, +1.49] 的漂移。
  所以本项目里 success 的 0.9 pt 以内、cit_p 的 0.7 pt 以内都不该解读（`confirm.md` §7.2）。
- 402 余额不足曾被 `except Exception` 吞成 `parse_error`，产出 2400+ 条静默垃圾 → 改为 `LLMError` 上抛 +
  开跑前 preflight + check 禁止坏终止类型。
- 支持 `STOP` 文件在批次边界暂停、按 qid 续跑（关机场景），坏行自动剔除备份。

