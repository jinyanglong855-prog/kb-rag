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
            order = [("a/项目状态/STATUS.md", 0), ("b/工作日志/x.md", 1)]
        else:
            order = [("b/工作日志/x.md", 1), ("a/项目状态/STATUS.md", 0)]
        return [{"path": p, "heading": "h", "text": "t", "score": 1.0, "idx": i}
                for p, i in order]

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


def _monopoly_routes(monkeypatch):
    """单文件霸榜场景：3 块来自同一状态页、1 块来自金标文件，各路排名一致。"""
    monkeypatch.setattr(rw, "rewrite_variants", lambda q, n=3: ["改写问法"])
    monkeypatch.setattr(rw, "hyde_document", lambda q: "")

    def fake_hybrid(route, k, vector_candidates):
        paths = ["s/项目状态/README.md"] * 3 + ["g/项目文件/README.md"]
        return [{"path": p, "heading": "h", "text": "t", "score": 1.0, "idx": i}
                for i, p in enumerate(paths)]

    monkeypatch.setattr(rw, "hybrid_query", fake_hybrid)


def test_fused_query_file_cap(monkeypatch):
    _monopoly_routes(monkeypatch)
    monkeypatch.delenv("KB_FUSION_FILE_CAP", raising=False)

    # 生产默认=2（2026-10-28 转正）：霸榜文件最多 2 席，第 3 席让位给金标文件
    fused = rw.fused_query("原问题", k=3)
    assert [h["path"] for h in fused] == ["s/项目状态/README.md", "s/项目状态/README.md",
                                          "g/项目文件/README.md"]

    # 显式置 0：回退无配额旧行为，霸榜文件占满 top3
    monkeypatch.setenv("KB_FUSION_FILE_CAP", "0")
    fused = rw.fused_query("原问题", k=3)
    assert [h["path"] for h in fused] == ["s/项目状态/README.md"] * 3

    # 非法值按生产默认处理
    monkeypatch.setenv("KB_FUSION_FILE_CAP", "abc")
    fused = rw.fused_query("原问题", k=3)
    assert [h["path"] for h in fused] == ["s/项目状态/README.md", "s/项目状态/README.md",
                                          "g/项目文件/README.md"]


def test_fused_query_authority_dedup(monkeypatch):
    """KB_FUSION_AUTHORITY=0：融合汇总去权威后排序跟随纯 RRF。

    三候选构造可区分的两种排序：纯 RRF 下 LOG（两路高排名）> STATUS（两路低排名）；
    乘权威（2.0 vs 0.6）后反转为 STATUS 第一。默认开=维持权威排序（既有测试固化）。
    """
    monkeypatch.setattr(rw, "rewrite_variants", lambda q, n=3: ["改写问法"])
    monkeypatch.setattr(rw, "hyde_document", lambda q: "")

    def fake_hybrid(route, k, vector_candidates):
        order = (["a/项目状态/STATUS.md", "b/工作日志/x.md", "c/notes.md"]
                 if route == "原问题"
                 else ["b/工作日志/x.md", "c/notes.md", "a/项目状态/STATUS.md"])
        idx = {"a/项目状态/STATUS.md": 0, "b/工作日志/x.md": 1, "c/notes.md": 2}
        return [{"path": p, "heading": "h", "text": "t", "score": 1.0, "idx": idx[p]}
                for p in order]

    monkeypatch.setattr(rw, "hybrid_query", fake_hybrid)

    monkeypatch.setenv("KB_FUSION_AUTHORITY", "0")
    fused = rw.fused_query("原问题", k=3)
    assert [h["path"] for h in fused] == ["b/工作日志/x.md", "a/项目状态/STATUS.md", "c/notes.md"]

    monkeypatch.delenv("KB_FUSION_AUTHORITY", raising=False)
    fused = rw.fused_query("原问题", k=3)
    assert [h["path"] for h in fused] == ["a/项目状态/STATUS.md", "c/notes.md", "b/工作日志/x.md"]
