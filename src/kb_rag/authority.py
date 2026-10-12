"""路径权威性加权：知识库有结构语义（现行状态 > 决策 > 日志 > 归档），
检索分数应尊重它。规则显式可解释，而不是黑盒重排。

按路径前缀匹配乘数；每条规则可回答"为什么这条内容该排前/后"。
"""

from functools import lru_cache

# (路径前缀, 乘数, 理由) —— 顺序即优先级，先命中先生效
AUTHORITY_RULES: list[tuple[str, float, str]] = [
    ("Shared/归档/", 0.35, "归档=历史留档，不承担当前状态"),
    ("/归档/", 0.35, "项目内归档同理"),
    ("/工作日志/", 0.6, "工作日志是过程记录，非权威事实源"),
    ("/CHANGELOG", 0.7, "变更历史，追溯用"),
    ("项目状态/STATUS.md", 2.0, "项目状态页=当前状态权威入口（收口规则落点）"),
    ("项目状态/README.md", 1.8, "项目状态主页=项目知识权威（知识沉淀层）"),
    ("kb-rag/项目文件/README.md", 1.5, "kb-rag 仓库 README=权威技术文档（项目文件区特例）"),
    ("/STATUS.md", 1.6, "其余状态页（根 STATUS 等）"),
    ("/决策区/", 1.3, "决策记录=现行有效取舍"),
    ("/README.md", 1.2, "项目主页=当前入口"),
    ("/项目文件/", 0.5, "真实文件区=原始素材/源码，知识结论以项目状态区为准"),
]


@lru_cache(maxsize=100_000)
def authority_multiplier(path: str) -> float:
    for prefix, mult, _reason in AUTHORITY_RULES:
        if prefix in path:
            return mult
    return 1.0


def apply(path: str, score: float) -> float:
    return score * authority_multiplier(path)
