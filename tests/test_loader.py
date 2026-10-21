"""语料收录规则回归：整目录排除（项目源码区=项目管理不入库，项目状态区=知识沉淀照收）、
排除树内的保留子树（校准账本）、项目状态区内部的深层数据区排除。"""

from kb_rag.loader import load_text_files


def _touch(root, rel, text="内容"):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_excluded_subtree_skips_md_and_source(tmp_path):
    _touch(tmp_path, "公司智能体/项目文件/deepseek-harness/README.md")
    _touch(tmp_path, "公司智能体/项目文件/admin-system/server/app.py")
    _touch(tmp_path, "公司智能体/项目状态/README.md")  # 知识沉淀照收
    _touch(tmp_path, "公司智能体/项目状态/交付记录.md")
    _touch(tmp_path, "kb-rag/项目文件/src/a.py")  # 其他项目的源码区不受影响
    _touch(tmp_path, "个人画像/项目状态/README.md")
    rels = sorted(rel for _, rel in load_text_files(tmp_path))
    assert rels == [
        "kb-rag/项目文件/src/a.py",
        "个人画像/项目状态/README.md",
        "公司智能体/项目状态/README.md",
        "公司智能体/项目状态/交付记录.md",
    ]


def test_excluded_subtree_is_prefix_match_not_substring(tmp_path):
    # 只排除该子树本身，同名前缀的兄弟目录（如 项目文件2）与别处的"公司智能体"片段不受影响
    _touch(tmp_path, "公司智能体/项目文件2/a.md")
    _touch(tmp_path, "Shared/公司智能体/项目文件/b.md")
    rels = sorted(rel for _, rel in load_text_files(tmp_path))
    assert rels == ["Shared/公司智能体/项目文件/b.md", "公司智能体/项目文件2/a.md"]


def test_excluded_whole_project_skips_corpus_and_wiki_node(tmp_path, monkeypatch):
    """整体废弃的项目：语料不收、摘要树不建节点（状态类问题不该被路由到死项目）。
    用合成名经 monkeypatch 验证机制，不依赖生产配置里的具体项目名。"""
    import kb_rag.loader as loader
    import kb_rag.wikitree as wt

    monkeypatch.setattr(loader, "EXCLUDED_SUBTREES", {"废弃项目"})
    monkeypatch.setattr(wt, "EXCLUDED_SUBTREES", {"废弃项目"})
    _touch(tmp_path, "废弃项目/项目状态/README.md")
    _touch(tmp_path, "废弃项目/项目文件/a.py")
    _touch(tmp_path, "在用项目/项目状态/README.md")
    assert [rel for _, rel in load_text_files(tmp_path)] == ["在用项目/项目状态/README.md"]
    assert [n.path for n in wt._discover(tmp_path)] == ["", "在用项目"]


def test_kept_subtree_survives_parent_exclusion(tmp_path, monkeypatch):
    """排除树内的保留子树（小红书 predictions 校准账本）照收；保留只放行该子树本身，
    同前缀兄弟目录不放行——放行口径与排除一样是严格路径段前缀。"""
    import kb_rag.loader as loader

    monkeypatch.setattr(loader, "EXCLUDED_SUBTREES", {"项目A/项目文件"})
    monkeypatch.setattr(loader, "KEPT_SUBTREES", {"项目A/项目文件/x/predictions"})
    _touch(tmp_path, "项目A/项目文件/x/predictions/p.md")
    _touch(tmp_path, "项目A/项目文件/x/predictions2/q.md")  # 同前缀兄弟：不放行
    _touch(tmp_path, "项目A/项目文件/x/archive/r.md")
    _touch(tmp_path, "项目A/项目状态/README.md")
    rels = sorted(rel for _, rel in load_text_files(tmp_path))
    assert rels == ["项目A/项目文件/x/predictions/p.md", "项目A/项目状态/README.md"]


def test_excluded_subtree_inside_status_area(tmp_path, monkeypatch):
    """数据区类排除可以落在项目状态区内部（候选池快照是数据查询对象，不是问答对象）；
    同名前缀的文件（数据区.md）不是子树，照收。"""
    import kb_rag.loader as loader

    monkeypatch.setattr(loader, "EXCLUDED_SUBTREES", {"项目B/项目状态/数据区"})
    _touch(tmp_path, "项目B/项目状态/数据区/候选池/a.md")
    _touch(tmp_path, "项目B/项目状态/数据区.md")
    _touch(tmp_path, "项目B/项目状态/产出区/b.md")
    rels = sorted(rel for _, rel in load_text_files(tmp_path))
    assert rels == ["项目B/项目状态/产出区/b.md", "项目B/项目状态/数据区.md"]


def test_index_artifact_dirs_skipped_by_prefix(tmp_path):
    """index/ 之外的索引产物目录（index_qwen 重建草稿、index_backup_* 旧备份）整体跳过；
    只按目录段判，文件名里的 index_ 前缀不受影响。"""
    _touch(tmp_path, "p/项目文件/index_backup_bge_20261017/bm25_meta.json")
    _touch(tmp_path, "p/项目文件/index_qwen/meta.json")
    _touch(tmp_path, "p/项目文件/src/index_utils.py")
    rels = sorted(rel for _, rel in load_text_files(tmp_path))
    assert rels == ["p/项目文件/src/index_utils.py"]
