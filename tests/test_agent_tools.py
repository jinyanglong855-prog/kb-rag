"""agent 检索工具的读侧过滤测试：_scoped 纯函数（不依赖索引/嵌入）。"""

from kb_rag.agent import _scoped


def _hit(path: str) -> dict:
    return {"path": path, "heading": "h", "text": "t", "score": 0.1}


HITS = [
    _hit("公司智能体/项目状态/a.md"),
    _hit("私有区/画像.md"),
    _hit("kb-rag/项目状态/b.md"),
]


def test_none_allowed_means_no_filter():
    assert _scoped(HITS, 5, None) == HITS


def test_filter_keeps_only_allowed_top_dirs():
    out = _scoped(HITS, 5, {"公司智能体"})
    assert [h["path"] for h in out] == ["公司智能体/项目状态/a.md"]


def test_star_passes_all():
    assert len(_scoped(HITS, 5, {"*"})) == 3


def test_truncate_after_filter():
    # 受限路径：先过滤再截断，配额留给可见内容
    out = _scoped(HITS, 1, {"公司智能体", "私有区"})
    assert len(out) == 1
    # 不受限路径：由调用方保证恰好 k 条，原样返回
    assert _scoped(HITS, 1, {"*"}) == HITS
