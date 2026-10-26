"""分析卡层（步骤 4/4 试点）：每文件一张 LLM 分析卡进语料，命中卡片回读原文。

Anthropic Contextual Retrieval 的文件级降本版（其报告：检索失败率 -35%嵌入 /
-49%+BM25 / -67%+精排）：LLM 通读整份文件，产出"能回答什么问题 + 关键事实 +
关联"的分析卡；卡片作为该文件的一个附加块进索引——口语问题先命中卡片，ask
装配层再回读原文进上下文，回答永远来自原文而非 AI 转述。

与改写层/摘要树同款纪律：
- 缓存按文件 SHA-1 指纹（INDEX_DIR/cards.json，损坏回退 .bak）：文件没变不重生
  成——同文件永远得到同一张卡，块哈希与检索逐位可复现；API 失败保留旧卡（宁旧
  勿缺）、同一文件连续失败 3 次退避跳过，绝不阻塞建索引。
- 卡片是索引产物不是知识条目：只写 INDEX_DIR/cards.json，不走 writer、不占 add
  配额、不进摘要树；试点在隔离 INDEX_DIR 进行，生产索引零接触（增量入库暂停至
  验收）。
- 隐私边界：卡片生成会把文件内容发给 LLM API（构建期批量导出，与 ask 随单次
  问题带片段出域性质不同）。2026-10-21 深夜"语料永不出域"硬约束下的例外：用户
  于 2026-10-26 拍板分析卡走 DeepSeek API 全库铺开（卡片按指纹缓存，文件不变
  不重发）；企业落地可把 LLM_BASE_URL 指向本机 OpenAI 兼容后端（语料不出机器）。
- 开关语义：默认开（2026-10-26 全库铺开），KB_CARDS=0 关闭——关掉只停"生成与
  纳入"，已进索引的卡片块随文件变更自然衰减（不做静默销毁）。命中裁决只在 ask
  装配层，检索排序与评测口径不动。试点模式（--pilot N）固定清单；全库模式
  （--all）动态跟随语料，新增文件自动建卡、删除文件自动清卡。
- 提示词注入面留档：文件内容原样进生成提示词，构造性文件可产出"万能卡"挤占
  检索名额；卡文本不进回答上下文、回读是原文，影响限于检索排序（语料本地可控，
  低危）。
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

from openai import OpenAI

from .config import (CHUNK_SIZE, INDEX_DIR, KB_PATH, LLM_API_KEY, LLM_BASE_URL,
                     LLM_MODEL, PARENT_MAX_CHARS, validate_kb_path)
from .loader import load_text_files

CARD_HEADING = "分析卡（AI 生成）"
MAX_CARD_CHARS = CHUNK_SIZE * 2  # 卡片块硬上限：控制单块长度，超限截断（提示词已要求 ≤600 字）
MAX_CONTENT_CHARS = 24_000  # 送 LLM 的文件内容上限（deepseek 64k token 上下文的安全占比）
READBACK_CHARS = PARENT_MAX_CHARS  # 卡片命中回读原文的预算，与父块同款
MAX_FAILURES = 3  # 同一文件同一内容连续生成失败次数上限：超过即退避，不再反复烧 LLM

CARD_PROMPT = """你是知识库的检索索引员。为下面这份文档写一张"分析卡"，让语义检索在用户
用口语提问时也能命中这份文档。分析卡固定四节：

## 这是什么
一句话：文档主题 + 属于哪个项目/领域。

## 能回答的问题
列 5-8 个答案就在本文档里的用户问题：口语问法与文档用语两种表述都要，正反都要
（如"还开着吗"与"已停止"）；涉及数字/版本/参数的问题单独列。

## 关键事实
列文档里的关键结论、数字、状态、名称、专有名词与别名，一行一个，检索关键词要全。

## 关联
一句话：与哪些其他文档/项目/主题相关。

要求：总共不超过 600 字；只依据下方文档内容，不要臆测未展示的部分（超长文档
只展示了前半部分时，不要对未展示内容下结论）；只输出分析卡正文，不要解释。

文档路径：{path}
文档内容：
{content}"""


def enabled() -> bool:
    """卡片层总开关：默认开（2026-10-26 全库铺开，用户拍板）；KB_CARDS=0 关。"""
    return os.environ.get("KB_CARDS", "1") == "1"


def cards_file() -> Path:
    return Path(INDEX_DIR) / "cards.json"


def _load() -> dict:
    """读卡片缓存；主文件损坏回退 .bak（最近一次完好缓存），双双不可读时按空缓存
    处理并打 _corrupt 标记——ensure_cards 会告警且不覆盖写回（用空缓存覆盖损坏文件
    = 销毁现场，selection 再也无法恢复）。"""
    p = cards_file()
    for path in (p, p.with_suffix(".json.bak")):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and isinstance(data.get("meta"), dict) \
                    and isinstance(data.get("cards"), dict):
                return data
    if p.exists() or p.with_suffix(".json.bak").exists():
        return {"meta": {}, "cards": {}, "_corrupt": True}
    return {"meta": {}, "cards": {}}


def _save(data: dict) -> None:
    payload = {k: v for k, v in data.items() if not k.startswith("_")}
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    cur = cards_file()
    if cur.exists():  # 留 .bak：主文件此后损坏时可回退到最近一次完好缓存
        try:
            json.loads(cur.read_text(encoding="utf-8"))
            shutil.copy2(cur, cur.with_suffix(".json.bak"))
        except (json.JSONDecodeError, OSError):
            pass
    tmp = cur.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, cur)  # 原子落盘：断电不留半个 cards.json


def _llm(prompt: str) -> str:
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=120, max_retries=2)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def pilot_selection(root: Path, n: int = 50) -> list[str]:
    """试点文件选取（确定性）：12 题评测金标文件优先，其余按项目轮询、项目内按
    文件体积降序——既保证评测可测出卡片效果，又保证各项目覆盖均匀。"""
    files = load_text_files(root)
    rel_set = {rel for _, rel in files}
    golden: list[str] = []
    try:
        from .evals import EVAL_SET  # 惰性导入：cards 在 indexer 构建路径上，避免模块环

        cases = json.loads((Path(root) / EVAL_SET).read_text(encoding="utf-8"))["cases"]
        for c in cases:
            for e in c["expected_any"]:
                if e in rel_set and e not in golden:
                    golden.append(e)
    except Exception:
        pass  # 评测集缺失不阻塞试点：退化为纯轮询选取
    by_proj: dict[str, list[tuple[int, str]]] = {}
    for f, rel in files:
        if rel not in golden:
            by_proj.setdefault(rel.split("/", 1)[0], []).append((f.stat().st_size, rel))
    for q in by_proj.values():
        q.sort(reverse=True)
    queues = [by_proj[k] for k in sorted(by_proj)]
    rest: list[str] = []
    while len(golden) + len(rest) < n and any(queues):
        for q in queues:
            if q and len(golden) + len(rest) < n:
                rest.append(q.pop(0)[1])
    return (golden + rest)[:n]


def generate_card(root: Path, rel: str) -> str:
    text = (Path(root) / rel).read_text(encoding="utf-8", errors="ignore")
    card = _llm(CARD_PROMPT.format(path=rel, content=text[:MAX_CONTENT_CHARS])).strip()
    if not card:
        raise RuntimeError("LLM 返回空内容")
    return card[:MAX_CARD_CHARS]


def _target_rels(meta: dict, file_hashes: dict[str, str]) -> list[str]:
    """本轮应服务的文件清单：全库模式=当前全部语料文件（新增文件自动建卡、
    删除文件自动清卡）；试点模式=缓存的固定 selection。"""
    if meta.get("selection_mode") == "all":
        return sorted(file_hashes)
    return meta.get("selection") or []


def _needs(rel: str, fh: str | None, cur: dict) -> bool:
    """该文件本轮是否需要（重）试生成。双指纹语义（复验 P1）：
    card_hash=卡片实际描述的内容（新鲜判定用它）；file_hash=最近一次尝试时的内容
    （退避计数用它）。失败条目只动 file_hash/errors，card 与 card_hash 不动——
    旧卡继续如实对应旧内容，下轮仍判过期、照常重试，不会"伪装成已同步"。"""
    if fh is None:
        return False  # 文件已删：无可生成
    if cur.get("card_hash", cur.get("file_hash")) == fh and cur.get("card"):
        return False  # 新鲜：卡与当前内容一致
    if cur.get("file_hash") == fh and cur.get("errors", 0) >= MAX_FAILURES:
        return False  # 同一内容连续失败到上限：退避
    return True


def ensure_cards(root: Path, file_hashes: dict[str, str], force: bool = False,
                 quiet: bool = False) -> tuple[list[dict], dict]:
    """生成/更新缺失与过期卡片，返回 (卡片块列表, 统计)。

    只为 selection 内文件服务；掉出 selection（重选/移出试点）与已删除文件的卡
    随之清理，缓存不残留孤儿卡。生成失败保留旧卡并计失败数（宁旧勿缺 + 退避），
    错误留痕不阻塞建索引。"""
    data = _load()
    if data.get("_corrupt"):
        print(f"[cards] ⚠ 卡片缓存损坏且备份不可读（{cards_file()}）：本轮按无卡构建，"
              "重跑 kb-rag cards --pilot N 可重建", file=sys.stderr, flush=True)
    meta = data.setdefault("meta", {})
    sel = _target_rels(meta, file_hashes)
    kept: dict = {rel: c for rel, c in (data.get("cards") or {}).items() if rel in set(sel)}
    generated, errors = 0, []
    for rel in sel:
        fh = file_hashes.get(rel)
        if fh is None:  # 文件已删：卡片随之失效
            kept.pop(rel, None)
            continue
        cur = kept.get(rel) or {}
        if not force and not _needs(rel, fh, cur):
            continue
        try:
            kept[rel] = {"file_hash": fh, "card_hash": fh, "card": generate_card(root, rel)}
            generated += 1
            if not quiet:
                print(f"[cards] {rel}", file=sys.stderr, flush=True)
        except Exception as e:
            errors.append(f"{rel}: {e}")
            # 只更新 file_hash（退避计数用）与 errors；card/card_hash 不动：
            # 旧卡继续如实对应旧内容，下轮仍判过期照常重试，绝不伪装成已同步
            kept[rel] = {**cur, "file_hash": fh, "errors": cur.get("errors", 0) + 1}
    meta.update({"model": LLM_MODEL, "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
    data["cards"] = kept
    if not data.get("_corrupt"):
        _save(data)
    chunks = [{"path": rel, "heading": CARD_HEADING, "text": c["card"], "card": 1}
              for rel, c in sorted(kept.items()) if c.get("card")]
    return chunks, {"generated": generated, "covered": len(chunks), "errors": errors}


def pending(file_hashes: dict[str, str]) -> bool:
    """selection 内是否存在本轮会重试的缺卡/过期卡（纯缓存对比，零 LLM）——供增量
    路径的 clean 短路判断：语料没变但卡片层待更新时仍要继续构建。退避中的文件
    不触发重建（重试也只会再次失败）。"""
    data = _load()
    meta = data.get("meta") or {}
    cards = data.get("cards") or {}
    return any(_needs(rel, file_hashes.get(rel), cards.get(rel) or {})
               for rel in _target_rels(meta, file_hashes))


def resolve_hits(hits: list[dict], budget: int | None = None) -> list[dict]:
    """ask 装配层：卡片命中 → 回读原文；同文件已有原文命中时卡片让位（不双占名额）。

    预算与父块同款（软上限：原文不足预算即全文）。回读按 path 直读磁盘——scope
    过滤在命中时已按同一 path 判定，回读不引入越权面；文件已删/不可读时该命中
    丢弃（宁缺毋假）。无卡片命中时原样返回，零开销。"""
    if not any(h.get("card") for h in hits):
        return hits
    if budget is None:
        budget = READBACK_CHARS
    root = Path(KB_PATH)
    orig_paths = {h["path"] for h in hits if not h.get("card")}
    out: list[dict] = []
    for h in hits:
        if not h.get("card"):
            out.append(h)
            continue
        if h["path"] in orig_paths:
            continue  # 原文已在上下文：卡片使命完成
        try:
            text = (root / h["path"]).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        out.append({**h, "text": text[:budget], "card_readback": True})
    return out


def show(arg: str) -> str:
    """查看已生成卡片：数字=前 N 张，其他=路径子串过滤。"""
    data = _load()
    cards = data.get("cards") or {}
    if data.get("_corrupt"):
        return f"卡片缓存损坏（{cards_file()}）：重跑 kb-rag cards --pilot N 重建"
    if not cards:
        return "尚无分析卡（kb-rag cards --pilot N 生成）"
    if arg.isdigit():
        items = sorted(cards.items())[:int(arg)]
        head = f"前 {len(items)}/{len(cards)} 张："
    else:
        items = [(rel, c) for rel, c in sorted(cards.items()) if arg in rel]
        head = f"匹配 '{arg}' {len(items)} 张："
    body = "\n\n".join(f"=== {rel} ===\n{c['card']}" for rel, c in items if c.get("card"))
    return f"{head}\n\n{body}"


def run_pilot(n: int = 0, regen: bool = False, all_mode: bool = False) -> str:
    """生成入口（持构建锁串行——与 index/add 并发时对 cards.json 的读改写会互相
    撕缓存）：all_mode=全库模式（全部语料文件建卡，日后新增文件自动建卡）；
    否则试点模式（评测金标优先+项目轮询选 n 个）。按文件指纹增量，缓存永久复用。"""
    if not LLM_API_KEY:
        raise SystemExit("请在 .env 里填 LLM_API_KEY（分析卡生成需要 LLM）")
    root = validate_kb_path()
    from .indexer import _acquire_build_lock, scan_file_hashes  # 惰性：避免模块环

    lock_fd, holder = _acquire_build_lock()
    if lock_fd is None:
        return f"另一索引构建进行中（{holder}）：本次卡片生成未执行，稍后重跑即可。"
    try:
        file_hashes = scan_file_hashes(root)
        data = _load()
        meta = data.setdefault("meta", {})
        lines: list[str] = []
        if all_mode:
            meta["selection_mode"] = "all"
            meta.pop("selection", None)
            _save(data)
            lines.append(f"全库模式：{len(file_hashes)} 个语料文件全部建卡（新增文件自动跟进）")
        elif regen or not meta.get("selection") or (n and len(meta["selection"]) != n):
            sel = pilot_selection(root, n or 50)
            meta["selection"] = sel
            meta["selection_mode"] = "pilot"
            _save(data)
            lines.append(f"已选取 {len(sel)} 个试点文件（评测金标优先+项目轮询）")
        sel = _target_rels(meta, file_hashes)
        chunks, stats = ensure_cards(root, file_hashes, force=regen)
        after = _load().get("cards") or {}  # ensure 内部已重写缓存：按新快照统计剩余缺口
        retry = sum(1 for rel in sel if _needs(rel, file_hashes.get(rel), after.get(rel) or {}))
        backoff = sum(1 for rel in sel
                      if not _needs(rel, file_hashes.get(rel), after.get(rel) or {})
                      and not (after.get(rel) or {}).get("card"))  # 退避中的缺卡文件
        lines.append(f"试点清单 {len(sel)} 文件；卡片覆盖 {stats['covered']}"
                     f"（本轮生成 {stats['generated']}，待重试 {retry}，退避暂缺 {backoff}）")
        lines.append(f"卡片清单：{cards_file()}")
        if stats["errors"]:
            lines.append(f"生成失败 {len(stats['errors'])}：\n  " + "\n  ".join(stats["errors"]))
        lines.append("\n样张（前 2 张）：\n" + "\n\n".join(
            f"=== {c['path']} ===\n{c['text']}" for c in chunks[:2]))
        return "\n".join(lines)
    finally:
        lock_fd.close()  # 关闭即释放 flock
