"""语料收录规则回归：整目录排除（项目源码区=项目管理不入库，项目状态区=知识沉淀照收）。"""

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
