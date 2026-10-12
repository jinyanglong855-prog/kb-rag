"""agent 检索工具测试：_scoped 过滤纯函数 + read_file 路径净化/scope 前置 + 图拒绝 + 失败审计。"""

import pytest

from kb_rag import audit, writer
from kb_rag.agent import _scoped, _tool_graph_query, _tool_read_file, run_tool


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


def test_graph_rejects_restricted_caller():
    # 权限矩阵"图拒绝"项的自动化回归（此前只有手动验证）
    msg = _tool_graph_query("任何问题", allowed={"公司智能体"})
    assert "拒绝" in msg and "vector/hybrid/status" in msg


def test_graph_star_passes_scope_gate():
    # 全范围角色过了 scope 闸门，走到"图未建立"检查（隔离环境无 lightrag 产物）
    msg = _tool_graph_query("任何问题", allowed={"*"})
    assert "未建立" in msg


def test_read_file_traversal_is_shape_rejected():
    msg = _tool_read_file("公共区/../私有区/x.md", allowed={"*"})
    assert "路径不合规" in msg  # 原始串上拦，不许借首段伪装越权


def test_read_file_scope_check_comes_before_existence():
    # 越界且不存在的文件 → scope 拒绝（不向受限角色泄漏文件是否存在）
    msg = _tool_read_file("私有区/根本不存在.md", allowed={"公共区"})
    assert msg.startswith("scope 拒绝")
    assert "公共区" in msg  # 拒绝信息列出允许范围，零上下文可自纠


def test_read_file_reads_in_scope(tmp_path):
    vault = writer.KB_PATH
    (vault / "公共区").mkdir(exist_ok=True)
    (vault / "公共区/可读.md").write_text("正文内容", encoding="utf-8")
    assert _tool_read_file("公共区/可读.md", allowed={"公共区"}) == "正文内容"


def test_read_file_missing_in_scope():
    (writer.KB_PATH / "公共区").mkdir(exist_ok=True)
    assert "文件不存在" in _tool_read_file("公共区/没有.md", allowed={"*"})


def test_run_tool_failure_is_audited():
    # 隔离环境无 vectors.npy → 检索必抛异常；失败必须先落审计再抛（不留无痕探测通道）
    with pytest.raises(Exception):
        run_tool("vector_search", caller="local", question="触发失败", k=1)
    rows = audit.tail(kind="read", limit=5)
    assert any(r.get("ok") is False and "vector_search" in r.get("tool", "") for r in rows)
