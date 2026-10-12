"""测试环境固定：本文件在任何测试模块导入 kb_rag 之前执行。

kb_rag.config 在导入时读环境变量（KB_PATH/INDEX_DIR/KB_SCOPES），所以
必须在首个 kb_rag 导入前设好——全部指向临时目录，测试永不触碰真实总库
与真实 index/（2026-10-11 索引覆盖事故的教训：toy 测试必须隔离 INDEX_DIR）。
"""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="kbrag-tests-"))

VAULT = _TMP / "vault"
(VAULT / "公共区").mkdir(parents=True)
(VAULT / "私有区").mkdir()
(VAULT / "README.md").write_text(
    "---\ntype: vault\nupdated: 2026-10-12\nstatus: active\nlifecycle: active\n---\n\n# 测试库\n",
    encoding="utf-8",
)

os.environ["KB_PATH"] = str(VAULT)
os.environ["INDEX_DIR"] = str(_TMP / "index")
os.environ["KB_SCOPES"] = str(_TMP / "scopes.json")
os.environ.pop("KB_WRITE_SCOPE", None)
os.environ.pop("KB_CALLER", None)
