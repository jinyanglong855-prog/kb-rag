"""查询改写层回归固化：缓存确定性、API 失败优雅回退、RRF 融合×权威乘数。

不打真实 LLM API：monkeypatch rewrite._llm / hybrid_query，秒级完成。
"""

import json

import kb_rag.rewrite as rw


def test_variants_cached_deterministic_and_fallback(monkeypatch):
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return '["变体一问法", "变体二问法", 123, "ok"]'  # 非字符串项应被过滤

    monkeypatch.setattr(rw, "_llm", fake_llm)
    v1 = rw.rewrite_variants("创作台还开着吗")
    assert v1 == ["变体一问法", "变体二问法", "ok"]  # 非字符串项(123)被过滤
    assert len(calls) == 1

    # 第二次问同一问题：走缓存，LLM 再也不被调（逐位可复现的根据）
    monkeypatch.setattr(rw, "_llm", lambda p: (_ for _ in ()).throw(RuntimeError("API 挂了")))
    v2 = rw.rewrite_variants("创作台还开着吗")
    assert v2 == v1 and len(calls) == 1

    # 换个新问题 + API 失败：回退空列表（fused_query 退化为原问题单路，不报错）
    assert rw.rewrite_variants("全新问题") == []
    assert rw.hyde_document("全新问题") == ""


def test_fused_query_rrf_with_authority(monkeypatch):
    monkeypatch.setattr(rw, "rewrite_variants", lambda q, n=3: ["改写问法"])
    monkeypatch.setattr(rw, "hyde_document", lambda q: "")

    def fake_hybrid(route, k, vector_candidates):
        if route == "原问题":  # 仅示意：两路返回同一批候选、排名不同
            order = ["a/项目状态/STATUS.md", "b/工作日志/x.md"]
        else:
            order = ["b/工作日志/x.md", "a/项目状态/STATUS.md"]
        return [{"path": p, "heading": "h", "text": "t", "score": 1.0} for p in order]

    monkeypatch.setattr(rw, "hybrid_query", fake_hybrid)
    fused = rw.fused_query("原问题", k=2)
    # RRF：STATUS 1/(60+1)+1/(60+2)=0.0325×2.0；日志 1/(60+2)+1/(60+1)=0.0325×0.6
    # 两路排名互换后 RRF 基分相同，权威乘数（2.0 vs 0.6）决定最终排序
    assert fused[0]["path"] == "a/项目状态/STATUS.md"
    assert fused[1]["path"] == "b/工作日志/x.md"
    assert fused[0]["score"] > fused[1]["score"]


def test_cache_file_is_valid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(rw, "CACHE", tmp_path / "rewrite_cache.json")
    monkeypatch.setattr(rw, "_llm", lambda p: '["x"]')
    rw.rewrite_variants("问题A")
    data = json.loads((tmp_path / "rewrite_cache.json").read_text(encoding="utf-8"))
    assert any("variants" in v for v in data.values())
