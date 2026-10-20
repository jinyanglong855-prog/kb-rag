"""检索提准第一轮（2026-10-17）的回归固化：权威先验融合 + 候选池默认值。

融合逻辑改动过一次曾零测试（对抗审查 P2），本文件防止再次静默破坏。
模型不真实加载：monkeypatch get_model 返回固定打分的假模型，秒级完成。
"""

import inspect
import math

import kb_rag.rerank as rerank_mod
from kb_rag.rerank import _sigmoid, rerank


def test_sigmoid_monotonic_and_stable():
    scores = [-30.0, -6.7, -1.0, 0.0, 1.0, 3.0, 30.0]
    sigmas = [_sigmoid(s) for s in scores]
    assert all(a < b for a, b in zip(sigmas, sigmas[1:]))  # 严格单调，同乘数组内保序的前提
    assert 0.0 < sigmas[0] and sigmas[-1] < 1.0  # 端点不溢出，乘权威乘数后仍落在 (0, 2] 量级
    assert math.isclose(_sigmoid(0.0), 0.5)


def test_rerank_fuses_authority_prior(monkeypatch):
    class FakeModel:
        def rerank(self, question, docs):
            return [5.0, 4.0, 3.0]  # 纯语义分：第 1 篇最高

    monkeypatch.setattr(rerank_mod, "get_model", lambda: FakeModel())
    hits = [
        {"path": "x/工作日志/a.md", "heading": "h", "text": "t"},  # ×0.6：语义最高但权威最低
        {"path": "x/项目状态/STATUS.md", "heading": "h", "text": "t"},  # ×2.0
        {"path": "x/项目状态/README.md", "heading": "h", "text": "t"},  # ×1.8
    ]
    ranked = rerank("q", hits)
    assert [h["path"] for h in ranked] == [
        "x/项目状态/STATUS.md", "x/项目状态/README.md", "x/工作日志/a.md",
    ]
    # 融合公式逐条可验：sigmoid(语义分) × 权威乘数；原始 rerank_score 保留
    for h, raw in zip(hits, [5.0, 4.0, 3.0]):
        assert h["rerank_score"] == round(raw, 4)


def test_hybrid_default_candidate_pool_is_50():
    from kb_rag.hybrid import hybrid_query

    default = inspect.signature(hybrid_query).parameters["vector_candidates"].default
    assert default == 50  # 20 池实测漏 2 个评测金标（35/50 池才进），降回须先过评测闸门


def test_embed_normalized_keeps_positions(monkeypatch):
    """向量必须落回原位置：错位会让整个索引无声地"向量 i ≠ 块 i"。"""
    import numpy as np

    import kb_rag.search as search_mod

    def fake_embed(texts):
        return [[float(len(t)), 1.0] for t in texts]  # 向量第一维 = 文本长度，可反查来源

    monkeypatch.setattr(search_mod, "embed", fake_embed)
    texts = ["aaaaa", "b", "ccc", "dddddddd", "ee"]  # 乱序长度 5,1,3,8,2
    out = search_mod._embed_normalized(texts, batch=2)

    # 位置 i 的向量来自 texts[i]（归一化前第一维等于长度）
    recovered = out[:, 0] / out[:, 1]
    assert np.allclose(recovered, [5, 1, 3, 8, 2])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert search_mod._embed_normalized([], batch=2).shape == (0, 1024)
