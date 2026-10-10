"""kb-rag 知识插件的 MCP server（stdio）——DSH 知识插件接入点。

上游调度框架 DeepSeek Harness（DSH，Node/TS，"一切皆插件"）通过 stdio
MCP server 消费外部工具：配置 {transport:'stdio', serverName, command,
args, env} 后，本进程暴露的全部工具按 mcp__<serverName>__<rawName>
命名空间化，形如 mcp__kb__kb_manifest。DSH 侧接入配置示例（契约的一部分）：

```yaml
mcpServers:
  kb:
    transport: stdio
    serverName: kb
    command: uv
    args: ["--directory", "/Users/yljin/Documents/项目文件/kb-rag/项目文件", "run", "kb-rag-mcp"]
```

首次接入先调 kb_manifest 获取契约（项目清单/别名/写入约定/索引新鲜度），
再决定用哪个检索工具。KB_PATH 无需 env 传入（config 自行解析仓库 .env）；
若宿主不继承 HOME 等环境变量导致嵌入模型缓存找不到，再在 env 里显式补齐。

stdio 协议下 stdout 是协议通道，本模块不做任何 print；模块顶部不执行
检索/IO，import 无副作用，运行时校验集中在 main()。
"""

import json

# mcp 2.x 将 FastMCP 更名为 MCPServer（mcp.server.fastmcp 已移除），构造/装饰器/run 接口同构
from mcp.server.mcpserver import MCPServer

from .agent import run_tool
from .ask import ask
from .config import validate_kb_path
from .manifest import build_manifest
from .writer import write_entry

mcp = MCPServer("kb-rag")


@mcp.tool()
def kb_manifest() -> str:
    """插件自描述契约：先调本工具了解可用项目、别名、写入约定与索引新鲜度，再决定用哪个检索工具。"""
    return json.dumps(build_manifest(), ensure_ascii=False, indent=2)


@mcp.tool()
def kb_ask(question: str, k: int = 5) -> str:
    """综合问答：检索 top-k 后由 LLM 生成带引用的回答，状态类问题自动走结构层，口语简称自动消解；需要 LLM_API_KEY。要直接得到答案而不是自己翻检索结果时用。"""
    return ask(question, k)


@mcp.tool()
def kb_vector_search(question: str, k: int = 5) -> str:
    """向量语义检索：问题语义模糊、说不出精确关键词时用，按含义相近召回，不依赖字面命中。"""
    return run_tool("vector_search", question=question, k=k)


@mcp.tool()
def kb_hybrid_search(question: str, k: int = 8) -> str:
    """BM25+向量混合检索：问题含项目名/人名/精确专有名词时用，字面与语义双路召回。"""
    return run_tool("hybrid_search", question=question, k=k)


@mcp.tool()
def kb_status_lookup(question: str) -> str:
    """结构层目录路由：问某项目的现状/进度/卡点/下一步时用，直接返回该项目状态页整文件。"""
    return run_tool("status_lookup", question=question)


@mcp.tool()
def kb_graph_query(question: str, mode: str = "mix") -> str:
    """知识图谱多跳检索：问跨项目关系、对比、多跳关联（"A 和 B 有什么关系"、"哪些项目都涉及 X"）时用；mode ∈ mix/local/global/hybrid。"""
    return run_tool("graph_query", question=question, mode=mode)


@mcp.tool()
def kb_read_file(path: str) -> str:
    """按库内相对路径读整个文件：前面工具的结果被截断、需要全文时用；拒绝越界路径。"""
    return run_tool("read_file", path=path)


@mcp.tool()
def kb_write(content: str, rel_path: str, caller: str = "dsh") -> str:
    """写入知识库唯一入口：frontmatter 必须含 type/updated/status/lifecycle（lifecycle ∈ active/experimental/deprecated/archived/generated），仅收 .md；写入后自动增量入库、立即可检索；拒绝 项目文件/ 目录与越界路径。"""
    return json.dumps(write_entry(content, rel_path, caller), ensure_ascii=False, indent=2)


def main() -> None:
    """kb-rag-mcp 脚本入口：运行时校验后以 stdio 传输启动。"""
    validate_kb_path()  # 失败即 SystemExit（中文提示），符合预期
    mcp.run()  # stdio 是默认传输
