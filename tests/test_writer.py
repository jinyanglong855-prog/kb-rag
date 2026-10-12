"""writer 写路径测试：路径/内容校验、scope 拒绝、审计 schema（kind 字段）。

索引同步被 monkeypatch 成 no-op——写路径逻辑不依赖真实嵌入；
读侧过滤逻辑在 test_agent_tools.py 用纯函数覆盖。
"""

import json

import pytest

from kb_rag import audit, writer

GOOD = (
    "---\ntype: note\nupdated: 2026-10-12\nstatus: active\nlifecycle: active\n---\n\n"
    "# 标题\n\n内容。\n"
)


@pytest.fixture(autouse=True)
def _no_sync(monkeypatch):
    monkeypatch.setattr(writer, "sync_incremental", lambda: {"mode": "skipped"})


@pytest.fixture(autouse=True)
def _clean_audit():
    writer.AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer.AUDIT_PATH.write_text("", encoding="utf-8")
    yield


def _audit_records() -> list[dict]:
    return [json.loads(l) for l in writer.AUDIT_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_rejects_absolute_and_traversal():
    assert not writer.write_entry(GOOD, "/etc/passwd.md")["ok"]
    assert not writer.write_entry(GOOD, "../outside.md")["ok"]
    assert not writer.write_entry(GOOD, "a/./b.md")["ok"]
    assert not writer.write_entry(GOOD, "a//b.md")["ok"]


def test_rejects_forbidden_segments_and_non_md():
    assert not writer.write_entry(GOOD, "项目文件/README.md")["ok"]
    assert not writer.write_entry(GOOD, ".obsidian/x.md")["ok"]
    assert not writer.write_entry(GOOD, "公司智能体/notes.txt")["ok"]


def test_rejects_missing_frontmatter():
    assert not writer.write_entry("# 无 frontmatter\n", "公共区/a.md")["ok"]
    bad = GOOD.replace("lifecycle: active", "lifecycle: forever")
    assert not writer.write_entry(bad, "公共区/a.md")["ok"]


def test_scope_rejects_out_of_bounds_write():
    rules = {"default": {"read": ["*"], "write": ["*"]},
             "callers": {"finance": {"read": ["公司智能体"], "write": ["公司智能体"]}}}
    from kb_rag import scopes
    p = writer.AUDIT_PATH.parent / "scopes_test.json"
    p.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")
    scopes._load.cache_clear()
    import os
    old = os.environ.get("KB_SCOPES")
    os.environ["KB_SCOPES"] = str(p)
    try:
        scopes._load.cache_clear()
        r = writer.write_entry(GOOD, "私有区/画像.md", caller="finance")
        assert not r["ok"] and "scope" in r["error"]
        assert not (writer.KB_PATH / "私有区/画像.md").exists()
    finally:
        if old:
            os.environ["KB_SCOPES"] = old
        else:
            os.environ.pop("KB_SCOPES", None)
        scopes._load.cache_clear()


def test_success_write_and_audit_schema():
    r = writer.write_entry(GOOD, "公共区/笔记.md", caller="finance")
    assert r["ok"]
    assert (writer.KB_PATH / "公共区/笔记.md").exists()
    records = _audit_records()
    assert len(records) == 1
    rec = records[0]
    assert rec["kind"] == "write" and rec["ok"] is True and rec["caller"] == "finance"
    assert rec["rel_path"] == "公共区/笔记.md" and "ts" in rec


def test_rejected_write_also_audited():
    writer.write_entry(GOOD, "项目文件/x.md", caller="finance")
    records = _audit_records()
    assert len(records) == 1 and records[0]["kind"] == "write" and records[0]["ok"] is False


def test_audit_consumer_reads_back():
    writer.write_entry(GOOD, "公共区/笔记.md", caller="tester")
    rows = audit.tail(limit=10)
    assert rows and rows[0]["kind"] == "write" and rows[0]["caller"] == "tester"
    assert list(audit.stats()) == ["tester/write"]
    assert "tester/write" in audit.format_stats()
    assert audit.tail(caller="nobody") == []
