# kb-rag · 本地知识库检索：六阶段演进

对本人 722 篇 Obsidian 知识库（4793 个语义块）做检索增强，从朴素 RAG 一路演进到 Agentic GraphRAG。每个阶段独立可讲：解决什么问题、为什么这个规模选这个组件。

## 六阶段路线

| 阶段 | 方案 | 要解决的问题 | 状态 |
| --- | --- | --- | --- |
| 0 | 朴素 RAG：BGE-zh 嵌入 + numpy 余弦 | 基线 | ✅ 当前 |
| 1 | 混合检索（BM25+向量）+ rerank；引入 Chroma | 精排与规模 | 待做 |
| 2 | Wiki 化层：目录层级摘要树（PageIndex/RAPTOR 思路） | "项目状态"类问题命中结构页而非归档 | 待做 |
| 3 | mini-GraphRAG：LLM 抽实体关系 → networkx → 多跳检索 → pyvis 可视化，对比 LightRAG | 跨项目多跳问题 | 待做 |
| 4 | Agentic RAG：各层检索做成 tools，LLM 自主调度 | 检索策略自适应 | 待做 |
| 5 | evals：自建单跳/多跳评测集，四层检索匹配率对比 | 可量化地证明每层的价值 | 待做 |

## 阶段 0 的设计决策（第一性原理）

- **722 篇 / 4793 块不用向量数据库**：numpy 余弦暴力检索即可毫秒级返回，先跑通再按真实规模选 Chroma/FAISS。
- **嵌入用本地 ONNX（fastembed + BGE-small-zh）**：零 API 成本，隐私不出本机。
- **Markdown 感知分块**：按标题层级切块并保留面包屑，而不是定长硬切。

## 阶段 0 已知缺陷（后续阶段的"before"样本）

问"小红书宠物项目现在卡在哪一步"，top3 命中的是 `Shared/归档/` 的历史快照，而正确答案在 `Codex/小红书宠物/STATUS.md`——朴素向量检索不理解**文档新鲜度与结构权威性**。这正是阶段 1（精排）/2（结构层）/3（图）要解决的。

## 用法

```bash
cp .env.example .env   # 填 KB_PATH（只读引用知识库，笔记不入库）
uv sync
uv run kb-rag index
uv run kb-rag query "小红书宠物项目现在卡在哪一步" -k 5
```
