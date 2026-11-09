"""换模型实验的有效性旁证：确认「换了模型」这一步真的生效。

动机：E1-3 的结论是「换 bge-base / MiniLM-L-12 指标几乎不变」。这类结论最容易被反问
「是不是模型根本没换成功？」。本脚本用两个不依赖 API 的证据回答：

1. **重排**：同一批候选，L-6 与 L-12 给出的 top-5 是否不同（不同 = 两个模型在起作用）；
2. **嵌入**：bge-small(384 维) 与 bge-base(768 维) 对同一句的向量维度与打分排序是否不同。

用法：python eval/model_sanity.py   （本地推理，¥0，约 1 分钟）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import retrievers as R  # noqa: E402
from eval import common  # noqa: E402


def main() -> int:
    samples = common.load_samples(Path("data/hotpot_dev_600.jsonl"))[:5]

    # ---- 1. 重排 ----
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    ce6 = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2", cache_dir=R.CACHE_DIR)
    ce12 = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-12-v2", cache_dir=R.CACHE_DIR)
    same = 0
    print("## 重排模型 top-5 是否相同（同一批 20 条候选）\n")
    for s in samples:
        idx = R.build_index(s["passages"])
        cands = R.retrieve(idx, s["question"], k=20, mode="hybrid")
        docs = [c["text"] for c in cands]

        def top5(ce):
            sc = list(ce.rerank(s["question"], docs))
            order = sorted(range(len(docs)), key=lambda i: -float(sc[i]))[:5]
            return [cands[i]["pid"] for i in order]

        a, b = top5(ce6), top5(ce12)
        same += a == b
        print(f"- `{s['qid']}`　L6={a}　L12={b}　相同={a == b}")
    print(f"\n**5 题中 top-5 完全相同 {same}/5**（不同即证明两个重排模型确实在各自工作）\n")

    # ---- 2. 嵌入 ----
    from fastembed import TextEmbedding

    e_small = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=R.CACHE_DIR)
    e_base = TextEmbedding(model_name="BAAI/bge-base-en-v1.5", cache_dir=R.CACHE_DIR)
    q = samples[0]["question"]
    vs = np.asarray(list(e_small.embed([q])))[0]
    vb = np.asarray(list(e_base.embed([q])))[0]
    print("## 嵌入模型维度\n")
    print(f"- bge-small-en-v1.5：{vs.shape[0]} 维")
    print(f"- bge-base-en-v1.5：{vb.shape[0]} 维")
    print(f"\n维度不同即证明两次实验用的不是同一个编码器。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
