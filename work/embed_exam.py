"""嵌入模型考卷：多臂对比检索准确率，为嵌入模型定版供证据（2026-10-21）。

四步：
  questions  12 道人工验证题 + N 道合成题（DeepSeek 从抽样块出题，金标=来源文件/块）
  corpus     考卷子集语料：每题 BM25/bge 双路 top-25 难负例 + 金标块 + 随机背景
  embed      各臂对同一子集嵌文档与问题（fastembed 本地 / OpenAI 兼容 API / llama-server）
  score      线上同款口径打分：vector@k、hybrid@8（RRF+权威）、hybrid+rerank@5

子集而非全库：考卷嵌入是抛式产物（正式重建在定版后），且各臂必须同一语料才可比。
局限（留档口径）：难负例来自 bge/BM25 视角，新模型的独有干扰可能未入池——
随机背景 2000 块缓解；臂间相对排名不受影响，绝对值略偏乐观。

用法:
  uv run python work/embed_exam.py questions [--n 48]
  uv run python work/embed_exam.py corpus
  uv run python work/embed_exam.py embed --arm bge
  uv run python work/embed_exam.py embed --arm q06q
  uv run python work/embed_exam.py embed --arm q8b-gguf --api-base http://127.0.0.1:8081/v1 --api-model qwen3-emb --api-batch 32
  uv run python work/embed_exam.py score
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # 模型已全部缓存，防联网校验卡死（2026-10-17 教训）

import kb_rag  # noqa: F401,E402  # 副作用：BLAS 单线程 + FASTEMBED_CACHE_PATH 钉仓库

import numpy as np  # noqa: E402

from kb_rag.config import INDEX_DIR, KB_PATH, validate_kb_path  # noqa: E402
from kb_rag.hybrid import _tokenize  # noqa: E402

STATE = Path(__file__).resolve().parent / "embed_exam"
QFILE = STATE / "questions.json"
CORPUS = STATE / "corpus.jsonl"
ARMS = STATE / "arms"
EVAL_SET = "kb-rag/项目状态/参考资料/eval_set.json"
OUT_REPORT = KB_PATH / "kb-rag/项目状态/参考资料" / f"embed_exam_{date.today().isoformat()}.json"

# Qwen3 官方检索指令（查询侧；文档侧不加前缀）
QWEN3_INSTRUCT = ("Instruct: Given a web search query, retrieve relevant passages "
                  "that answer the query\nQuery:{q}")
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

FASTEMBED_ARMS = {
    # 名: (fastembed 模型名, 文档批大小, 查询指令族)
    "bge": ("BAAI/bge-small-zh-v1.5", 64, "bge-zh"),
    "q06q": ("Qwen/Qwen3-Embedding-0.6B-Q", 4, "qwen3"),
}


def _load_questions() -> list[dict]:
    data = json.loads(QFILE.read_text(encoding="utf-8"))
    return data["questions"]


def _load_corpus() -> list[dict]:
    return [json.loads(line) for line in CORPUS.open(encoding="utf-8")]


def _norm(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-10)


# ---------------------------------------------------------------- questions

SYNTH_PROMPT = """你是检索评测集构建助手。下面是知识库中的一段内容及其路径、标题。
请以知识库真实用户的视角，写一个该内容能回答的自然问题。

要求：
1. 口语化提问，像一个用户会对检索系统说的话；
2. 不要照抄原文中的独特词组，换一种说法（同义改写），否则题目失去区分度；
3. 问题中不得出现文件名、路径、日期以外的编号式引用；
4. 答案必须能从这段内容得出，而不是需要外部知识；
5. 如果这段内容不适合出题（纯链接列表、元数据、无语义的代码片段、碎片表格），返回 {{"skip": true}}。

只输出 JSON：{{"question": "...", "answer": "（一句话答案，供人工抽检）"}}

路径: {path}
标题: {heading}
正文:
{text}"""


def _pick_chunks_for_questions(n: int) -> list[dict]:
    """分层抽样适合出题的块：项目状态 md / Shared md / 项目文件 md / 源码。"""
    chunks = json.loads((Path(INDEX_DIR) / "chunks.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)
    pools: dict[str, list[int]] = {"status": [], "shared": [], "projfile": [], "code": []}
    for i, c in enumerate(chunks):
        t = len(c["text"])
        if not (150 <= t <= 1500):
            continue
        if "/项目状态/" in c["path"] and c["path"].endswith(".md"):
            pools["status"].append(i)
        elif c["path"].startswith("Shared/") and c["path"].endswith(".md"):
            pools["shared"].append(i)
        elif "/项目文件/" in c["path"] and c["path"].endswith(".md"):
            pools["projfile"].append(i)
        elif "/项目文件/" in c["path"]:
            pools["code"].append(i)
    # 配比 ×3 过采样：LLM 判 skip 时换下一候选，不重跑
    quota = {"status": 0.55, "shared": 0.15, "projfile": 0.15, "code": 0.15}
    picked: list[int] = []
    for k, pool in pools.items():
        m = int(n * quota[k] * 3)
        idx = rng.choice(pool, size=min(m, len(pool)), replace=False)
        picked.extend(int(i) for i in idx)
    rng.shuffle(picked)
    return [chunks[i] | {"_idx": i} for i in picked[: n * 3]]


def _gen_one(client, model: str, c: dict) -> dict | None:
    for attempt in range(2):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": SYNTH_PROMPT.format(**c)}],
                response_format={"type": "json_object"},
                temperature=0.8,
            )
            d = json.loads(r.choices[0].message.content)
            if d.get("skip"):
                return None
            q = (d.get("question") or "").strip()
            if len(q) >= 6:
                return {"id": f"syn-{c['_idx']}", "type": "synthetic",
                        "question": q, "expected_any": [c["path"]],
                        "gold_chunk_key": f"{c['path']}#{c['heading']}",
                        "answer": (d.get("answer") or "").strip()}
            return None
        except Exception as e:
            if attempt == 1:
                print(f"  syn-{c.get('_idx')} 失败: {str(e)[:120]}", flush=True)
    return None


def cmd_questions(n: int) -> None:
    validate_kb_path()
    from openai import OpenAI
    from kb_rag.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

    verified = json.loads((KB_PATH / EVAL_SET).read_text(encoding="utf-8"))["cases"]
    cands = _pick_chunks_for_questions(n)
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    print(f"合成 {n} 题：候选 {len(cands)} 块（{LLM_MODEL} 出题）…", flush=True)
    with ThreadPoolExecutor(max_workers=6) as ex:
        outs = list(ex.map(lambda c: _gen_one(client, LLM_MODEL, c), cands))
    synth = [o for o in outs if o][:n]
    print(f"合成成功 {len(synth)}/{n}（skip/失败 {len(outs) - len(synth)}）")
    questions = verified + synth
    STATE.mkdir(exist_ok=True)
    QFILE.write_text(json.dumps(
        {"date": date.today().isoformat(), "n_verified": len(verified),
         "n_synthetic": len(synth), "questions": questions},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"题目落盘 {QFILE}")


# ---------------------------------------------------------------- corpus

def cmd_corpus(vec_top: int = 25, bm_top: int = 25, random_bg: int = 2000) -> None:
    validate_kb_path()
    import bm25s
    from fastembed import TextEmbedding

    questions = _load_questions()
    chunks = json.loads((Path(INDEX_DIR) / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(Path(INDEX_DIR) / "vectors.npy")  # bge 全库向量（难负例视角）

    qs = [q["question"] for q in questions]
    # BM25 难负例（全库索引，取全局块号）
    bm25 = bm25s.BM25.load(str(INDEX_DIR / "bm25"))
    toks = [_tokenize(q) for q in qs]
    bm_idx = bm25.retrieve(toks, k=bm_top)[0]
    # bge 向量难负例
    bge = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5", threads=8)
    qv = _norm(np.array(list(bge.embed(qs, batch_size=64)), dtype=np.float32))
    sim = qv @ vectors.T  # (nq, N)
    vec_idx = np.argsort(-sim, axis=1)[:, :vec_top]

    selected: set[int] = set()
    gold_meta: dict[str, list[int]] = {}
    for qi, q in enumerate(questions):
        s: set[int] = set()
        if q["type"] == "synthetic":
            key = q["gold_chunk_key"]
            s |= {i for i, c in enumerate(chunks)
                  if f"{c['path']}#{c['heading']}" == key}
        else:
            for e in q["expected_any"]:
                m = [i for i, c in enumerate(chunks) if e in c["path"]]
                if len(m) > 60:  # 宽金标（目录级）：按 bge 相似度取前 60，控制子集规模
                    order = np.argsort(-sim[qi, m])
                    m = [m[j] for j in order[:60]]
                s |= set(m)
        gold_meta[q["id"]] = sorted(s)
        selected |= s
        selected |= {int(i) for i in bm_idx[qi]}
        selected |= {int(i) for i in vec_idx[qi]}
        if qi % 20 == 0:
            print(f"  难负例收集 {qi + 1}/{len(questions)}，累计 {len(selected)} 块", flush=True)

    rng = np.random.default_rng(42)
    rest = np.array(sorted(set(range(len(chunks))) - selected))
    bg = rng.choice(rest, size=min(random_bg, len(rest)), replace=False)
    selected |= {int(i) for i in bg}

    keep = sorted(selected)
    STATE.mkdir(exist_ok=True)
    with CORPUS.open("w", encoding="utf-8") as f:
        for i in keep:
            c = chunks[i]
            f.write(json.dumps({"idx": i, "path": c["path"], "heading": c["heading"],
                                "text": c["text"]}, ensure_ascii=False) + "\n")
    (STATE / "corpus_meta.json").write_text(json.dumps(
        {"size": len(keep), "gold": gold_meta}, ensure_ascii=False), encoding="utf-8")
    print(f"子集语料 {len(keep)} 块 → {CORPUS}")


# ---------------------------------------------------------------- embed

def _client_embed(texts: list[str], base: str, model: str, key: str,
                  batch: int, dims: int | None, cache: Path, tag: str) -> np.ndarray:
    """OpenAI 兼容 /v1/embeddings（DashScope 兼容模式 / SiliconFlow / llama-server 通用）。"""
    from openai import OpenAI

    out_f = cache / f"{tag}.npy"
    if out_f.exists():  # 断点续嵌：读已有行数
        have = np.load(out_f)
        start = have.shape[0]
        vecs = [have] if start else []
    else:
        start, vecs = 0, []
    if start >= len(texts):
        return have
    client = OpenAI(api_key=key or "none", base_url=base, timeout=600, max_retries=2)
    t0 = time.time()
    part_f = cache / f"{tag}.part.npy"
    part: list[np.ndarray] = []
    for i in range(start, len(texts), batch):
        kw = {"model": model, "input": texts[i:i + batch]}
        if dims:
            kw["dimensions"] = dims
        r = client.embeddings.create(**kw)
        d = sorted(r.data, key=lambda x: x.index)
        part.append(np.array([x.embedding for x in d], dtype=np.float32))
        if len(part) * batch >= 512:  # 阶段落盘防进程被杀全损；512 粒度便于定位卡点
            vecs.append(np.vstack(part)); part = []
            np.save(out_f, np.vstack(vecs))
            print(f"  {tag}: {i + batch}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    if part:
        vecs.append(np.vstack(part))
    out = np.vstack(vecs)
    np.save(out_f, out)
    print(f"  {tag}: 完成 {out.shape} {time.time() - t0:.0f}s（{out.shape[0] / (time.time() - t0):.1f} 块/s）")
    return out


def cmd_embed(arm: str, api_base: str = "", api_model: str = "", api_key: str = "",
              api_batch: int = 16, api_dims: int | None = None) -> None:
    validate_kb_path()
    from fastembed import TextEmbedding

    corpus = _load_corpus()
    questions = _load_questions()
    docs = [f"{c['path']} {c['heading']}\n{c['text']}" for c in corpus]
    qs_raw = [q["question"] for q in questions]
    adir = ARMS / arm
    adir.mkdir(parents=True, exist_ok=True)

    def save(name: str, arr: np.ndarray, t0: float, kind: str) -> None:
        np.save(adir / f"{name}.npy", _norm(arr))
        print(f"  {arm}/{name}: {arr.shape} {time.time() - t0:.0f}s"
              f"（{arr.shape[0] / max(time.time() - t0, 1e-9):.1f} 条/s）", flush=True)

    if arm in FASTEMBED_ARMS:
        model_name, batch, family = FASTEMBED_ARMS[arm]
        m = TextEmbedding(model_name=model_name, threads=8)
        t0 = time.time()
        save("doc", np.array(list(m.embed(docs, batch_size=batch)), dtype=np.float32), t0, "doc")
        t0 = time.time()
        save("q_raw", np.array(list(m.embed(qs_raw, batch_size=batch)), dtype=np.float32), t0, "q")
        t0 = time.time()
        pref = (lambda q: BGE_ZH_QUERY_PREFIX + q) if family == "bge-zh" \
            else (lambda q: QWEN3_INSTRUCT.format(q=q))
        save("q_inst", np.array(list(m.embed([pref(q) for q in qs_raw], batch_size=batch)),
                                dtype=np.float32), t0, "q")
    else:
        assert api_base and api_model, "API 臂需 --api-base 与 --api-model"
        t0 = time.time()
        d = _client_embed(docs, api_base, api_model, api_key, api_batch, api_dims, adir, "doc")
        save("doc", d, t0, "doc")
        t0 = time.time()
        save("q_raw", _client_embed(qs_raw, api_base, api_model, api_key,
                                    max(api_batch, 16), api_dims, adir, "q_raw"), t0, "q")
        if "qwen3" in api_model.lower() or arm.startswith(("q8b", "q4b", "v4")):
            t0 = time.time()
            save("q_inst", _client_embed([QWEN3_INSTRUCT.format(q=q) for q in qs_raw],
                                         api_base, api_model, api_key,
                                         max(api_batch, 16), api_dims, adir, "q_inst"), t0, "q")
    meta = {"arm": arm, "model": api_model or FASTEMBED_ARMS.get(arm, ("?",))[0],
            "api_base": api_base, "dims": api_dims, "date": date.today().isoformat()}
    (adir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- score

def _path_hit(expected: list[str], paths: list[str]) -> bool:
    norm = lambda s: s.replace("／", "/")  # noqa: E731
    return any(any(norm(e) in norm(p) for p in paths) for e in expected)


def cmd_score() -> None:
    validate_kb_path()
    import bm25s

    corpus = _load_corpus()
    questions = _load_questions()
    gold_meta = json.loads((STATE / "corpus_meta.json").read_text(encoding="utf-8"))["gold"]
    n = len(corpus)
    paths = [c["path"] for c in corpus]
    keys = [f"{c['path']}#{c['heading']}" for c in corpus]
    # corpus_meta 里存的是全库块号（idx），检索排名用子集行号——先建映射
    pos_of_idx = {c["idx"]: p for p, c in enumerate(corpus)}

    # 子集 BM25（与线上同一分词；只在子集上建，用于 hybrid 臂）
    bm25 = bm25s.BM25()
    bm25.index([_tokenize(f"{c['path']} {c['heading']} {c['text']}") for c in corpus])
    qs_toks = [_tokenize(q["question"]) for q in questions]
    bm_rank = bm25.retrieve(qs_toks, k=min(50, n))[0]  # (nq, 50) 子集行号

    from fastembed.rerank.cross_encoder import TextCrossEncoder
    from kb_rag.authority import authority_multiplier as auth_apply
    from kb_rag.rerank import _sigmoid

    reranker = TextCrossEncoder(model_name="BAAI/bge-reranker-base", threads=1)

    arms = sorted(p.name for p in ARMS.iterdir() if (p / "doc.npy").exists())
    results: dict[str, dict] = {}
    print(f"臂: {arms}；题 {len(questions)}；子集 {n}")
    for arm in arms:
        adir = ARMS / arm
        doc = np.load(adir / "doc.npy")
        for qv_name in ("q_raw", "q_inst"):
            if not (adir / f"{qv_name}.npy").exists():
                continue
            qv = np.load(adir / f"{qv_name}.npy")
            name = arm if qv_name == "q_raw" else f"{arm}+inst"
            res: dict = {"per_q": {}}
            sims = qv @ doc.T  # (nq, n)
            for qi, q in enumerate(questions):
                order = np.argsort(-sims[qi])
                gold = {pos_of_idx[i] for i in gold_meta[q["id"]] if i in pos_of_idx}
                v_top = [i for i in order[:50]]
                rrf: dict[int, float] = {}
                for r, bi in enumerate(bm_rank[qi], 1):
                    rrf[int(bi)] = rrf.get(int(bi), 0.0) + 1.0 / (60 + r)
                for r, vi in enumerate(v_top, 1):
                    rrf[vi] = rrf.get(vi, 0.0) + 1.0 / (60 + r)
                hyb = sorted(rrf, key=lambda i: -auth_apply(paths[i]) * rrf[i])[:8]
                rr_docs = [f"{paths[i]} {corpus[i]['heading']}\n{corpus[i]['text']}" for i in hyb]
                rr_scores = list(reranker.rerank(q["question"], rr_docs))
                rr = [i for _, i in sorted(
                    zip([-_sigmoid(float(s)) * auth_apply(paths[i]) for i, s in zip(hyb, rr_scores)],
                        hyb))]  # 分高在前：负 sigmoid×权威 排序
                gset = gold if gold else set()
                def first(hit_set, seq):
                    for j, i in enumerate(seq, 1):
                        if i in hit_set:
                            return j
                    return None
                row = {
                    "gold_n": len(gold),
                    "v_rank": first(gset, order.tolist()),
                    "hyb_rank": first(gset, hyb),
                    "rr_rank": first(gset, rr),
                    "chunk_v_rank": None,
                }
                if q["type"] == "synthetic":
                    row["chunk_v_rank"] = first({keys.index(q["gold_chunk_key"])} if
                                                q["gold_chunk_key"] in keys else set(), order.tolist())
                res["per_q"][q["id"]] = row
            results[name] = res
            print(f"  {name} 打分完成", flush=True)

    def agg(per_q: dict, ids: list[str]) -> dict:
        # 金标 0 块的题（宽路径没匹配到任何块）不计分，避免误判失分
        rows = [per_q[i] for i in ids if i in per_q and per_q[i]["gold_n"] > 0]
        out: dict = {"n": len(rows)}
        for k, div in (("v_rank", 5), ("hyb_rank", 8), ("rr_rank", 5)):
            vals = [r[k] for r in rows if r[k] is not None]
            out[f"{k}@{div}内"] = f"{len(vals)}/{len(rows)}"
            hits = [r[k] for r in rows if r[k] is not None and r[k] <= div]
            out[f"{k}@{div}命中"] = round(len(hits) / len(rows), 3)
            mrr = [1 / r[k] for r in rows if r[k] is not None and r[k] <= 10]
            out[f"{k}_MRR@10"] = round(sum(mrr) / len(rows), 3) if rows else 0.0
        cv = [r["chunk_v_rank"] for r in rows if r["chunk_v_rank"] is not None]
        if cv:
            out["chunk@5命中"] = round(sum(1 for v in cv if v <= 5) / len(rows), 3)
        return out

    v_ids = [q["id"] for q in questions if q["type"] != "synthetic"]
    s_ids = [q["id"] for q in questions if q["type"] == "synthetic"]
    summary = {name: {"verified": agg(r["per_q"], v_ids), "synthetic": agg(r["per_q"], s_ids),
                      "all": agg(r["per_q"], v_ids + s_ids)}
               for name, r in results.items()}

    print("\n===== 考卷结果（file 级；v=纯向量 hyb=混合RRF+权威 rr=+精排）=====")
    for name, s in summary.items():
        print(f"\n[{name}]")
        for grp in ("verified", "synthetic", "all"):
            a = s[grp]
            print(f"  {grp:9} v@5={a['v_rank@5命中']:.0%}({a['v_rank@5内']})  "
                  f"hyb@8={a['hyb_rank@8命中']:.0%}({a['hyb_rank@8内']})  "
                  f"rr@5={a['rr_rank@5命中']:.0%}({a['rr_rank@5内']})  "
                  f"vMRR@10={a['v_rank_MRR@10']}"
                  + (f"  chunk@5={a['chunk@5命中']:.0%}" if "chunk@5命中" in a else ""))

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(json.dumps(
        {"date": date.today().isoformat(),
         "setup": {"questions": len(questions), "corpus": n,
                   "arms": {a: json.loads((ARMS / a / "meta.json").read_text(encoding="utf-8"))
                            for a in arms if (ARMS / a / "meta.json").exists()},
                   "口径": "子集考卷（难负例=bge/BM25 top25 双路 + 金标 + 随机2000）；"
                          "hyb=子集BM25+向量RRF池50×权威；rr=bge-reranker-base sigmoid×权威"},
         "summary": summary, "per_question": {k: v["per_q"] for k, v in results.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n留档: {OUT_REPORT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["questions", "corpus", "embed", "score"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--arm", default="")
    ap.add_argument("--api-base", default="")
    ap.add_argument("--api-model", default="")
    ap.add_argument("--api-key", default=os.environ.get("EMBED_API_KEY", ""))
    ap.add_argument("--api-batch", type=int, default=16)
    ap.add_argument("--api-dims", type=int, default=None)
    args = ap.parse_args()
    {"questions": lambda: cmd_questions(args.n), "corpus": cmd_corpus,
     "embed": lambda: cmd_embed(args.arm, args.api_base, args.api_model,
                                args.api_key, args.api_batch, args.api_dims),
     "score": cmd_score}[args.cmd]()


if __name__ == "__main__":
    main()
