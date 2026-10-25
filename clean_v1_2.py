# -*- coding: utf-8 -*-
"""
clean_v1_2.py — 政府公文生成模型 V1.2 数据清洗

用法（在 D:\\codes\\Policy_crawler 目录下）：
    python clean_v1_2.py
    # 或显式指定：
    python clean_v1_2.py --input data\\jsonl\\Guangdong_V1_full_2951.jsonl --output v1_2

原则（严格按任务书）：
  * 不修改、不覆盖、不删除 V1 原始数据；本脚本只读原始文件，输出全部写到 v1_2/
  * 只删「明显不适合作为政府公文训练样本」的数据：占位符 / 机械扩写 / 重复 / 真截断 / 异常结构
  * 正常政策引用、历史年份、历史文号、发布信息、公文套话 → 一律保留
  * 长 / 短本身都不是删除理由
  * 全程确定性（固定 seed=42），每条删除都可追溯到具体规则

输出：
  v1_2/V1_2_clean.jsonl              最终训练用数据（原始字段 + text 字段）
  v1_2/rejected.jsonl                被删数据：uid/标题/删除原因/命中规则/文本统计
  v1_2/cleaning_report.json          清洗报告（任务书第四节要求的全部统计）
  v1_2/samples_kept_20.jsonl         人工抽查：保留样本 20 条
  v1_2/samples_rejected_20.jsonl     人工抽查：删除样本 20 条（含关键片段）
  v1_2/mechanical_risk_top20.jsonl   机械扩写风险 TOP 20（含被保留的边界样本）
  v1_2/train.jsonl / val.jsonl / test.jsonl   文档级 90/5/5 划分（UID 零泄漏）
  v1_2/split_v1_2.json               划分留痕
"""

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

SEED = 42

# ===========================================================================
# A. 占位符检测
# 只打「模板残留」型占位符；正常历史文号（如 国发〔2023〕12号）不会命中。
# ===========================================================================
PLACEHOLDER_RULES = [
    # “待补充”是模板占位残留的标志词；“有待完善/亟待完善”是正常公文用语，不算
    (re.compile(r"待补充"), "placeholder_待补充"),
    (re.compile(r"[（(〔]\s*(待定稿|待填写|待完善|待添加)[\s〕）)]"), "placeholder_括号占位"),
    (re.compile(r"〔\s*(文件编号|文号|编号|日期|机关名称)\s*〕"), "placeholder_空括号占位"),
    (re.compile(r"[XxＸｘ]{4,}"), "placeholder_XXXX"),
    (re.compile(r"〔\s*[XxＸｘ]{2,}\s*〕"), "placeholder_文号X占位"),
    (re.compile(r"_{4,}"), "placeholder_下划线填空"),
]

# ===========================================================================
# E. 真截断检测（只判明显没写完的）
# ===========================================================================
TRUNC_TAIL_ENUM = re.compile(r"(一是|首先|第一)$")          # 以“一是”收尾、没有下文
TRUNC_HEADING_END = re.compile(r"^[（(][一二三四五六七八九十]+[）)][^。；：]{0,30}$")
TRUNC_COLON_END = re.compile(r"[：:，、]$")

# 落款行（有落款+日期的，一般不是截断）
SIGNOFF = re.compile(r"(人民政府|办公厅|厅|局|委员会|办公室)\s*$")
DATE_LINE = re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日")

# ===========================================================================
# F. 异常结构 / 乱码
# ===========================================================================
GARBLED = re.compile(r"[□■]{2,}|&\w+;|<[a-zA-Z/][^>]{0,40}>")
WEIRD_CHAR = re.compile(r"[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
                        r"a-zA-Z0-9\s，。；：？！、（）()《》〈〉“”‘’—…·\-〔〕/年月日.%％]+")

CN_NUM = "一二三四五六七八九十"


def split_sentences(text: str):
    """按句读切句，保留换行段。过滤过短碎片。"""
    parts = re.split(r"(?<=[。！？；])|\n", text)
    return [p.strip() for p in parts if p and p.strip()]


def normalize_sentence(s: str) -> str:
    """句级归一化：去标点/数字/空白，用于近似重复检测。"""
    s = re.sub(r"[\s，。；：？！、（）()《》“”‘’—…·\-〔〕]", "", s)
    s = re.sub(r"\d+", "", s)
    return s


def shingles(s: str, n: int = 4):
    return {hashlib.md5(s[i:i + n].encode("utf-8")).hexdigest()[:12]
            for i in range(0, max(0, len(s) - n + 1))}


def analyze(text: str) -> dict:
    """对单篇正文做全部检测，返回指标 + 命中规则。"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sents = split_sentences(text)
    stats = {
        "chars": len(text), "lines": len(lines), "sentences": len(sents),
        "placeholder_hits": [], "dup_sentences": 0, "repeated_sentence_ratio": 0.0,
        "longest_repeat_run": 1, "near_dup_sentences": 0, "near_dup_ratio": 0.0,
        "second_half_unique_ratio": 1.0, "shingle4_repeat_ratio_2nd": 0.0,
        "mechanical_risk": 0, "truncated": False, "trunc_reason": "",
        "structure_issues": [], "garbled_hits": 0,
    }
    if not text:
        stats["structure_issues"].append("空正文")
        return stats

    # ---- A 占位符 ----
    for pat, name in PLACEHOLDER_RULES:
        if pat.search(text):
            stats["placeholder_hits"].append(name)

    # ---- B 完全相同句子重复 ----
    # 法规引用句（《XX条例》《XX决定》等）的重复是正常公文表达，不参与重复统计
    citation = re.compile(r"《[^》]{2,40}》")
    norm_sents = [normalize_sentence(s) for s in sents if not citation.search(s)]
    norm_sents = [s for s in norm_sents if len(s) >= 6]  # 短句（标题/落款）不参与
    cnt = Counter(norm_sents)
    dup = sum(c - 1 for c in cnt.values() if c > 1)
    stats["dup_sentences"] = dup
    stats["repeated_sentence_ratio"] = round(dup / len(norm_sents), 3) if norm_sents else 0.0
    run = best = 1
    for i in range(1, len(norm_sents)):
        run = run + 1 if norm_sents[i] == norm_sents[i - 1] else 1
        best = max(best, run)
    stats["longest_repeat_run"] = best

    # ---- C 近似重复（只替换少量词、句法高度一致）----
    # 机械扩写的典型特征是“连续若干句只差一两个词”，
    # 因此对每句只与前 12 句做 difflib 相似度（控制成本，且覆盖连续扩写窗口）。
    from difflib import SequenceMatcher
    near_dup = 0
    win = 12
    # 只对“长句”判近似重复：条款枚举（（一）…的；（二）…的；）是正常公文结构，
    # 句子短、信息各不同；机械扩写的特征是“长句式整体循环、只换个别名词”。
    long_sents = [x for x in norm_sents if len(x) >= 25]
    for i in range(len(long_sents)):
        si = long_sents[i]
        for j in range(max(0, i - win), i):
            sj = long_sents[j]
            if abs(len(si) - len(sj)) > max(len(sj) * 0.2, 5):
                continue
            if SequenceMatcher(None, si, sj).ratio() >= 0.88:
                near_dup += 1
                break
    near_base = len(long_sents)
    stats["near_dup_sentences"] = near_dup
    stats["near_dup_ratio"] = round(near_dup / near_base, 3) if near_base else 0.0

    # ---- D 机械扩写风险（后半部分信息增量）----
    half = len(norm_sents) // 2
    second = norm_sents[half:]
    if len(second) >= 6:
        stats["second_half_unique_ratio"] = round(len(set(second)) / len(second), 3)
        sh = [h for s in second for h in shingles(s)]
        if sh:
            stats["shingle4_repeat_ratio_2nd"] = round(1 - len(set(sh)) / len(sh), 3)
    # 风险评分：重复句 + 近似重复 + 后半唯一率 + 后半 shingle 重复
    risk = 0.0
    risk += min(30, stats["repeated_sentence_ratio"] * 60)          # 最多 30
    risk += min(30, stats["near_dup_ratio"] * 60)                   # 最多 30
    if len(second) >= 6:
        risk += (1 - stats["second_half_unique_ratio"]) * 20        # 最多 20
        risk += stats["shingle4_repeat_ratio_2nd"] * 20             # 最多 20
    stats["mechanical_risk"] = round(risk, 1)

    # ---- E 真截断（只看结尾形态，不看长度）----
    last = lines[-1] if lines else ""
    has_signoff_or_date = bool(SIGNOFF.search(last) or DATE_LINE.search(last)
                               or (len(lines) >= 2 and (SIGNOFF.search(lines[-2])
                                                        or DATE_LINE.search(lines[-2]))))
    if TRUNC_COLON_END.search(last):
        stats["truncated"] = True
        stats["trunc_reason"] = f"以“{last[-1]}”突兀收尾: …{last[-30:]}"
    elif TRUNC_TAIL_ENUM.search(last):
        stats["truncated"] = True
        stats["trunc_reason"] = f"以“{last[-4:]}”列举开头后无下文"
    elif (TRUNC_HEADING_END.match(last) and not has_signoff_or_date):
        stats["truncated"] = True
        stats["trunc_reason"] = f"以孤立小标题收尾: {last}"

    # ---- F 异常结构 / 乱码 / 非公文混入 ----
    g = GARBLED.findall(text)
    stats["garbled_hits"] = len(g)
    if g:
        stats["structure_issues"].append(f"乱码/HTML残留 {len(g)} 处")
    weird = len(WEIRD_CHAR.findall(text))
    if len(text) > 200 and weird / len(text) > 0.03:
        stats["structure_issues"].append(f"非常规字符占比 {weird / len(text):.1%}")
    return stats


def decide(doc_stats: dict) -> list:
    """根据指标给出删除原因列表（空列表 = 保留）。阈值保守，宁漏勿错。"""
    reasons = []
    s = doc_stats
    if s["chars"] == 0:
        return ["空正文"]
    # A 占位符：命中模板残留类即删
    if s["placeholder_hits"]:
        reasons.append("占位符:" + ",".join(s["placeholder_hits"]))
    # B 重复句：机械生成程度才删
    if s["longest_repeat_run"] >= 5 or \
       (s["repeated_sentence_ratio"] >= 0.5 and s["dup_sentences"] >= 10
        and s["second_half_unique_ratio"] < 0.35):
        reasons.append(f"重复句(连续{s['longest_repeat_run']}句/重复率{s['repeated_sentence_ratio']:.0%})")
    # C 机械扩写（近似句式循环，只换名词）：不要求长度
    if s["near_dup_ratio"] >= 0.6 and s["near_dup_sentences"] >= 10:
        reasons.append(f"机械扩写(近似句式循环{s['near_dup_sentences']}句,"
                       f"占比{s['near_dup_ratio']:.0%})")
    # D 机械扩写（长 + 高重复/低信息增量）
    elif s["chars"] >= 1500 and s["mechanical_risk"] >= 60:
        reasons.append(f"机械扩写(风险{s['mechanical_risk']},"
                       f"近似重复率{s['near_dup_ratio']:.0%},"
                       f"后半唯一率{s['second_half_unique_ratio']:.0%})")
    # E 真截断
    if s["truncated"]:
        reasons.append(f"截断({s['trunc_reason']})")
    # F 异常结构
    if s["garbled_hits"] >= 3:
        reasons.append("乱码/HTML残留")
    for issue in s["structure_issues"]:
        if issue.startswith("非常规字符"):
            reasons.append("异常文本:" + issue)
    return reasons


# ===========================================================================
# 文档级近重复（同一政策不同版本/重复抓取），用于划分时聚到同一 split
# ===========================================================================
def doc_signature(text: str):
    norm = normalize_sentence(text)
    sh = shingles(norm, n=8)
    # 轻量 LSH：4 个 band，各取最小 hash 作桶键
    if not sh:
        return []
    sh = sorted(sh)
    bands = []
    for b in range(4):
        sample = sh[b::4][:40]
        bands.append(hashlib.md5("".join(sample).encode()).hexdigest()[:10])
    return bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="sft_v1/cleaned/guangdong_cleaned.jsonl")
    ap.add_argument("--output", default="v1_2")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--train-ratio", type=float, default=0.90)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    args = ap.parse_args()

    in_path, out = Path(args.input), Path(args.output)
    if not in_path.is_file():
        print(f"[错误] 找不到输入文件: {in_path.resolve()}")
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] 读取 {in_path}（只读，不修改原始数据）...")
    docs = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            mt = obj.get("mainText")
            if not isinstance(mt, list):
                mt = mt.splitlines() if isinstance(mt, str) else []
            if isinstance(obj.get("text"), str) and obj["text"].strip():
                obj["text"] = obj["text"]
            else:
                obj["text"] = "\n".join(x.strip().strip("\u3000") for x in mt if x.strip())
            docs.append(obj)
    print(f"  共 {len(docs)} 篇")

    print("[2/6] 逐篇检测（占位符/重复句/近似重复/机械扩写/截断/结构）...")
    kept, rejected = [], []
    all_stats = []
    for d in docs:
        st = analyze(d["text"])
        all_stats.append((d, st))
        reasons = decide(st)
        if reasons:
            rejected.append({
                "uid": d.get("UID", ""), "title": d.get("title", ""),
                "reasons": reasons,
                "text_stats": {k: st[k] for k in
                               ("chars", "sentences", "dup_sentences",
                                "repeated_sentence_ratio", "longest_repeat_run",
                                "near_dup_ratio", "mechanical_risk")},
                "excerpt": d["text"][:200] + ("……" if len(d["text"]) > 200 else ""),
            })
        else:
            kept.append(d)

    print(f"  保留 {len(kept)} / 删除 {len(rejected)} "
          f"（删除率 {len(rejected) / max(len(docs), 1):.1%}）")

    print("[3/6] 写 rejected.jsonl / V1_2_clean.jsonl ...")
    with open(out / "rejected.jsonl", "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "V1_2_clean.jsonl", "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("[4/6] 生成人工抽查样本与机械扩写 TOP20 ...")
    rng = random.Random(args.seed)
    def sample_view(d, st, reasons=None):
        v = {"uid": d.get("UID", ""), "title": d.get("title", ""),
             "chars": st["chars"], "mechanical_risk": st["mechanical_risk"],
             "excerpt_head": d["text"][:150], "excerpt_tail": d["text"][-150:]}
        if reasons:
            v["reasons"] = reasons
        return v
    rej_by_uid = {r["uid"]: r["reasons"] for r in rejected}
    kept_view = [sample_view(d, st) for d, st in all_stats if d.get("UID") not in rej_by_uid]
    rng.shuffle(kept_view)
    with open(out / "samples_kept_20.jsonl", "w", encoding="utf-8") as f:
        for v in kept_view[:20]:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    rej_sample = rejected[:]
    rng.shuffle(rej_sample)
    with open(out / "samples_rejected_20.jsonl", "w", encoding="utf-8") as f:
        for v in rej_sample[:20]:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    top20 = sorted((sample_view(d, st, rej_by_uid.get(d.get("UID")))
                    for d, st in all_stats),
                   key=lambda v: -v["mechanical_risk"])[:20]
    with open(out / "mechanical_risk_top20.jsonl", "w", encoding="utf-8") as f:
        for v in top20:
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    print("[5/6] 文档级划分 train/val/test（近重复文档强制同 split）...")
    # 近重复聚类：共享任一 band 桶的文档归入同一簇
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    bucket_owner = {}
    uid2doc = {d.get("UID", f"__{i}"): d for i, d in enumerate(kept)}
    for uid, d in uid2doc.items():
        for band in doc_signature(d["text"]):
            if band in bucket_owner:
                union(uid, bucket_owner[band])
            else:
                bucket_owner[band] = uid
    clusters = {}
    for uid in uid2doc:
        clusters.setdefault(find(uid), []).append(uid)
    cluster_list = list(clusters.values())
    rng.shuffle(cluster_list)
    n_total = len(uid2doc)
    n_val_target = int(n_total * args.val_ratio)
    n_test_target = int(n_total * args.val_ratio)
    train_u, val_u, test_u = set(), set(), set()
    # 整簇分配：小簇优先填 test/val，大簇留给 train，避免大簇挤占小集合
    for cl in sorted(cluster_list, key=len):
        if len(test_u) + len(cl) <= n_test_target or len(test_u) == 0 and cl == cluster_list[0]:
            test_u.update(cl)
        elif len(val_u) + len(cl) <= n_val_target or (len(val_u) == 0 and len(cl) <= max(n_val_target, 5)):
            val_u.update(cl)
        else:
            train_u.update(cl)
    # 兜底：test/val 为空时从 train 挪一簇
    if not test_u and train_u:
        lone = next(iter(train_u)); train_u.discard(lone); test_u.add(lone)
    if not val_u and train_u:
        lone = next(iter(train_u)); train_u.discard(lone); val_u.add(lone)
    overlap = (train_u & val_u) | (train_u & test_u) | (val_u & test_u)
    assert not overlap, f"UID 泄漏: {overlap}"
    for name, uids in (("train", train_u), ("val", val_u), ("test", test_u)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for uid in sorted(uids):
                f.write(json.dumps(uid2doc[uid], ensure_ascii=False) + "\n")
    with open(out / "split_v1_2.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "ratio": "90/5/5 文档级",
                   "train": sorted(train_u), "val": sorted(val_u),
                   "test": sorted(test_u),
                   "uid_overlap": len(overlap)}, f, ensure_ascii=False, indent=1)
    print(f"  train {len(train_u)} / val {len(val_u)} / test {len(test_u)}，UID overlap = 0 ✔")

    print("[6/6] cleaning_report.json ...")
    reason_counter = Counter()
    for r in rejected:
        for reason in r["reasons"]:
            reason_counter[reason.split("(")[0].split(":")[0]] += 1
    lengths = sorted(st["chars"] for _, st in all_stats if st["chars"] > 0)
    n = len(lengths)
    def pct(p):
        return lengths[min(n - 1, int(n * p))] if n else 0
    report = {
        "原始总数": len(docs),
        "保留数量": len(kept),
        "删除数量": len(rejected),
        "删除率": f"{len(rejected) / max(len(docs), 1):.2%}",
        "各删除原因数量": dict(reason_counter),
        "重复样本数量": reason_counter.get("重复句", 0),
        "重复句数量": sum(st["dup_sentences"] for _, st in all_stats),
        "机械扩写数量": reason_counter.get("机械扩写", 0),
        "占位符数量": reason_counter.get("占位符", 0),
        "截断数量": reason_counter.get("截断", 0),
        "异常短文本数量": sum(1 for _, st in all_stats if 0 < st["chars"] < 50),
        "异常结构数量": sum(1 for _, st in all_stats if st["structure_issues"]),
        "保留数据统计": {
            "样本数": len(kept),
            "平均正文长度": round(statistics.mean(lengths), 1) if n else 0,
            "P50": pct(0.5), "P90": pct(0.9), "P95": pct(0.95), "P99": pct(0.99),
            "最大长度": lengths[-1] if n else 0,
        },
        "split": {"train": len(train_u), "val": len(val_u),
                  "test": len(test_u), "uid_overlap": 0},
    }
    with open(out / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"\n完成。输出目录: {out.resolve()}")
    print("下一步：请把 cleaning_report.json、samples_kept_20.jsonl、"
          "samples_rejected_20.jsonl、mechanical_risk_top20.jsonl 的内容发给我，"
          "人工确认清洗没有过严/过松，再进入训练阶段。")


if __name__ == "__main__":
    main()
