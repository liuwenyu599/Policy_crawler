#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
政府公文生成能力验收 V1.3 版（推理层强化）
============================================================

V1.3 更新（推理层改造，不动 V1.2 数据与 LoRA）：

目的：把"模型能力问题"与"推理层问题"分开，
      推理能修的问题不让用户看见。

1. 当前日期由程序注入（system prompt 写入今天日期）
2. 当前文号由程序生成：LLM 被禁止撰写本文件文号；
   若模型仍写了文头号，后处理剥离；
   程序根据落款单位推导代字（粤府办/国办发/…），
   按 〔当前年份〕序号 生成并插入标题之后
3. 历史政策文号允许正常引用，只做格式规范化
   （O/０/１/［］ 等明显污染机械修复，年份不合理不猜）
4. 生成期退化早停：连续重复片段 / 尾部 4-gram 多样性
   < 0.28 → 立即停止，标记 D 级，不再写到上限
5. 异常超长提前停：超过 4000 token 且多样性 < 0.50 → 提前停
   （健康公文多样性 >= 0.75，余量充分）
6. < 800 字异常短生成：标记并自动用采样解码重试一次，
   取评级更优者
7. 双轨评级：原始评级 = 模型真实能力；
   交付评级 = 推理层修复后的可用性，PASS/FAIL 以此为准

V1.2 的判定哲学是"可直接使用的完整文章"：
    - 文号是公文正常组成部分：格式规范即可，允许省略，
      只禁止占位式文号（XX 号、待补充）
    - 成文日期是必需组成部分：写具体日期即可（使用者按需替换），
      只禁止占位式日期（XXXX 年 XX 月 XX 日）
    - "本意见自 2026 年 X 月 X 日起施行"是正常公文表述，不再误报
    - 正常引用已有政策、历史文号、历史年份，不视为问题

真正拦截的硬伤（D 级，不可用，需重新生成）：
    1. 占位符（待补充 / XXXX 年 XX 月 / XX 号 / ×× 等）
    2. 退化循环（连续重复片段、尾部 4-gram 多样性崩塌）
    3. 机械重复（同一句出现 4 次以上，法条引用句除外）
    4. 截断（达到长度上限且结尾悬空 / 书名号未闭合）
    5. 乱码（替换符、私用区字符、连续 □）

需人工完善（B 级）：
    - 缺落款单位或成文日期
    - 缺标题或主送机关
    - 正文过短（< 800 字）
    - 主题相关性弱
    - 结尾悬空但未达长度上限

A 级：结构完整、无任何硬伤 —— 可直接使用。

通过标准：
    A 级比例 >= 85% 且 D 级为 0 → PASS

运行：

    # 生成 + 验收（V1.2 模型）
    python sft_v1/evaluation/run_generation_acceptance_v12.py --model v12

    # 对已有生成结果重新评分（不加载模型，可用于对比 V5 老结果）
    python sft_v1/evaluation/run_generation_acceptance_v12.py \
        --score-only sft_v1/evaluation/generation_acceptance/generation_results.jsonl

支持模型：base / v1 / v11 / v12 / all

输出目录：
    v1_2/evaluation/generation_acceptance/
        generation_results_v12.jsonl
        generation_acceptance_report_v12.txt
        H01_行政检查.txt ... H20_法治建设.txt   # 纯净正文，可直接使用
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

import re
import json
import time
import argparse
import gc
import unicodedata
from datetime import date
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path("/home/lwy/Policy_crawler")

BASE_MODEL = "/home/lwy/Qwen2.5-14B-Instruct"

ADAPTERS = {
    "v1": (
        "/home/lwy/Policy_crawler/"
        "sft_v1/checkpoints/real_sft_v1_qlora_full"
    ),
    "v11": (
        "/home/lwy/Policy_crawler/"
        "sft_v1/checkpoints/real_sft_v11_qlora_full"
    ),
    "v12": (
        "/home/lwy/Policy_crawler/"
        "v1_2/checkpoints/real_sft_v12_lora_full"
    ),
}

OUT_DIR = (
    PROJECT_ROOT
    / "v1_2"
    / "evaluation"
    / "generation_acceptance"
)


# ============================================================
# 2. 生成配置
# ============================================================

MAX_NEW_TOKENS = 8192
REPETITION_PENALTY = 1.10
DO_SAMPLE = False
NO_REPEAT_NGRAM_SIZE = 6

# 推理层保护参数
LONG_GEN_TOKENS = 4000       # 超过此 token 数转入严格监控
EARLY_STOP_CHECK_EVERY = 64  # 每生成 64 token 检查一次退化
SHORT_RETRY_MIN_CHARS = 800  # 低于此字数标记并重试一次

TODAY = date.today()
TODAY_STR = f"{TODAY.year}年{TODAY.month}月{TODAY.day}日"


# ============================================================
# 3. System Prompt
#    目标：完整、可直接使用的公文
# ============================================================

SYSTEM_PROMPT = """
你是一名政府机关公文起草人员。

请根据用户给出的主题和文种，直接起草一份正式、完整、规范的政府公文。

【整体目标】

输出一篇结构完整、可以直接使用的公文，包括：
标题、主送机关、正文、落款单位、成文日期。

【结构要求】

1. 第一行为标题，完整、居中语义、不加句号。
2. 标题后为主送机关，以冒号结尾。
3. 正文按照该文种的通行结构展开：
   - 通知：通知缘由、主要事项、工作要求。
   - 意见：总体要求、重点任务、保障措施。
   - 实施方案：总体要求、工作目标、主要任务、
     实施步骤、保障措施。
   - 工作方案：总体要求、主要任务、工作安排、
     工作机制、保障措施。
4. 正文结束后另起一行写落款单位全称。
5. 落款单位下一行写成文日期，
   成文日期使用今天：{TODAY}。
6. 禁止为本文件撰写文号：
   不要写"粤府办〔2026〕12号"这类本文件文号行，
   文号由程序在生成后自动生成并插入。
   引用其他已发布政策文件的文号可以正常写。

【绝对禁止】

1. 禁止任何占位符：
   "待补充"、"待完善"、"XXXX年XX月XX日"、
   "XX号"、"XX〔2026〕XX号"、"××"、"　　号"。
2. 禁止机械扩写：不得为了增加篇幅，
   用同义句反复表达同一项工作。
3. 禁止重复：同一句子不得在全文出现两次以上；
   不得复制前面已经出现的段落。
4. 禁止堆砌同义条目，例如连续生成大量
   "加强……建设""加强……监督"式空泛条目。
5. 禁止在半句话、半个条款、半个标题上停止；
   最后一项必须写完整，全文自然收尾。
6. 禁止声称本文件已经正式发布、已印发执行。
   （"本意见自2026年X月X日起施行"是正常表述，允许使用。）
7. 禁止输出"（本文有删减）""（略）"等转载/节选标记，
   草稿必须是完整全文。
8. 禁止输出分析、解释、写作说明、免责声明，
   只输出公文正文本身。

【允许】

1. 正常引用已存在的法律法规、政策文件名称及其真实文号。
2. 正常出现历史年份、历史政策背景、工作目标年份。
3. 使用公文惯用语和固定表述。

请直接输出公文正文。
"""


# ============================================================
# 4. 测试任务（H01-H20，与历次验收一致）
# ============================================================

TESTS = [
    {"id": "H01", "category": "行政检查", "genre": "通知",
     "prompt": "请起草一份关于规范行政检查工作的通知。"},
    {"id": "H02", "category": "行政执法", "genre": "实施方案",
     "prompt": "请起草一份关于规范行政执法行为的实施方案。"},
    {"id": "H03", "category": "政务服务", "genre": "工作方案",
     "prompt": "请起草一份关于提升政务服务效能的工作方案。"},
    {"id": "H04", "category": "依法行政", "genre": "意见",
     "prompt": "请起草一份关于推进依法行政工作的意见。"},
    {"id": "H05", "category": "规范性文件", "genre": "通知",
     "prompt": "请起草一份关于加强行政规范性文件管理的通知。"},
    {"id": "H06", "category": "基层治理", "genre": "实施方案",
     "prompt": "请起草一份关于推进基层治理现代化的实施方案。"},
    {"id": "H07", "category": "法治政府", "genre": "工作方案",
     "prompt": "请起草一份关于推进法治政府建设的工作方案。"},
    {"id": "H08", "category": "行政复议", "genre": "意见",
     "prompt": "请起草一份关于进一步规范行政复议工作的意见。"},
    {"id": "H09", "category": "公共法律服务", "genre": "实施方案",
     "prompt": "请起草一份关于加强基层公共法律服务体系建设的实施方案。"},
    {"id": "H10", "category": "政务公开", "genre": "通知",
     "prompt": "请起草一份关于进一步推进政务公开工作的通知。"},
    {"id": "H11", "category": "行政执法监督", "genre": "实施方案",
     "prompt": "请起草一份关于加强行政执法监督工作的实施方案。"},
    {"id": "H12", "category": "政府法律顾问", "genre": "意见",
     "prompt": "请起草一份关于加强政府法律顾问工作的意见。"},
    {"id": "H13", "category": "政府行政行为", "genre": "工作方案",
     "prompt": "请起草一份关于规范政府行政行为的工作方案。"},
    {"id": "H14", "category": "数字政府", "genre": "实施方案",
     "prompt": "请起草一份关于推进数字政府建设的实施方案。"},
    {"id": "H15", "category": "政务数据", "genre": "通知",
     "prompt": "请起草一份关于加强政务数据管理工作的通知。"},
    {"id": "H16", "category": "人工智能", "genre": "意见",
     "prompt": "请起草一份关于规范政府部门人工智能应用的意见。"},
    {"id": "H17", "category": "行政审批", "genre": "实施方案",
     "prompt": "请起草一份关于深化行政审批制度改革的实施方案。"},
    {"id": "H18", "category": "营商环境", "genre": "工作方案",
     "prompt": "请起草一份关于进一步优化营商环境的工作方案。"},
    {"id": "H19", "category": "权力监督", "genre": "意见",
     "prompt": "请起草一份关于加强行政权力运行监督的意见。"},
    {"id": "H20", "category": "法治建设", "genre": "通知",
     "prompt": "请起草一份关于加强法治政府建设的通知。"},
]

# 主题相关性关键词
RELEVANCE_KEYWORDS = {
    "行政检查": ["行政检查", "检查"],
    "行政执法": ["行政执法", "执法"],
    "政务服务": ["政务服务", "服务效能", "办事"],
    "依法行政": ["依法行政", "依法"],
    "规范性文件": ["规范性文件", "文件管理"],
    "基层治理": ["基层治理", "基层"],
    "法治政府": ["法治政府", "法治"],
    "行政复议": ["行政复议", "复议"],
    "公共法律服务": ["公共法律服务", "法律服务"],
    "政务公开": ["政务公开", "公开"],
    "行政执法监督": ["执法监督", "行政执法"],
    "政府法律顾问": ["法律顾问", "律师"],
    "政府行政行为": ["行政行为", "行政"],
    "数字政府": ["数字政府", "数字", "数字化"],
    "政务数据": ["政务数据", "数据"],
    "人工智能": ["人工智能", "智能"],
    "行政审批": ["行政审批", "审批"],
    "营商环境": ["营商环境", "营商"],
    "权力监督": ["权力", "监督"],
    "法治建设": ["法治", "依法"],
}


# ============================================================
# 5. 检测器 —— 硬伤（D 级）
# ============================================================

# ---------- 5.1 占位符 ----------

PLACEHOLDER_PATTERNS = [
    r"待补充",
    r"待完善",
    r"待填写",
    r"待确定",
    r"待添加",
    # 占位式日期：XXXX年XX月XX日 / XX月XX日 / X年X月
    r"[XxＸｘ×]{2,}年[XxＸｘ×]{1,}月",
    r"[XxＸｘ×]{1,}月[XxＸｘ×]{1,}日",
    r"某年某月",
    # 占位式文号：XX号 / XX〔2026〕XX号 / 〔　〕
    r"[XxＸｘ×]{2,}〔",
    r"〔\s*[XxＸｘ×\s]*\d{0,4}\s*[XxＸｘ×\s]*〕\s*[XxＸｘ×]*号",
    r"[XxＸｘ×]{2,}号",
    r"××+",
    r"ＸＸ+",
    # 长下划线 / 空书名号括号
    r"_{3,}",
    r"＿{2,}",
    r"《\s*》",
    r"〔\s*〕",
]


def detect_placeholders(text: str) -> Dict[str, Any]:
    matches = []
    for pattern in PLACEHOLDER_PATTERNS:
        matches.extend(re.findall(pattern, text))
    # 归一化展示
    matches = list(dict.fromkeys(
        m if isinstance(m, str) else "".join(m)
        for m in matches
    ))
    return {"hit": len(matches) > 0, "matches": matches[:10]}


# ---------- 5.2 退化循环 ----------

def detect_degeneration(text: str) -> Dict[str, Any]:
    """
    两类退化：
    a) 连续重复片段：(.{8,60}) 连续出现 3 次以上
    b) 局部 4-gram 多样性崩塌（抓 H03 式的逐字变化循环）
    """
    m = re.search(r"(.{8,60}?)\1{2,}", text, flags=re.DOTALL)
    if m:
        return {
            "hit": True,
            "kind": "连续重复片段",
            "evidence": m.group(1)[:60] + " ×" + str(
                len(m.group(0)) // max(len(m.group(1)), 1)
            ),
        }

    # 全文分窗检查 4-gram 多样性。
    # 校准基准：健康生成 >= 0.75，真实公文最低 0.41，
    # 退化样本 0.23-0.26。
    # 为避免长文枚举条款（责任分工、阶段安排等）
    # 单窗口下坠造成误报，要求"持续退化"：
    # >= 3 个窗口低于 0.35，或单窗口极端低于 0.28。
    plain = re.sub(r"\s+", "", text)
    window = 300
    worst = 1.0
    low_windows = 0
    for start in range(0, max(len(plain) - window, 1), 150):
        seg = plain[start:start + window]
        grams = [seg[i:i + 4] for i in range(len(seg) - 3)]
        if not grams:
            continue
        diversity = len(set(grams)) / len(grams)
        worst = min(worst, diversity)
        if diversity < 0.35:
            low_windows += 1
    if low_windows >= 3 or worst < 0.28:
        return {
            "hit": True,
            "kind": "局部文本多样性崩塌",
            "evidence": f"4-gram 多样性最低 {worst:.2f}，"
                        f"低多样性窗口 {low_windows} 个"
                        f"（健康生成 >= 0.75，真实公文 >= 0.4）",
        }
    return {"hit": False}


# ---------- 5.3 机械重复（句级） ----------

def normalize_sentence(s: str) -> str:
    s = re.sub(r"[\s，。；、：：（）()《》〈〉\"'“”‘’]", "", s)
    return s


def detect_repetition(text: str) -> Dict[str, Any]:
    sentences = re.split(r"[。；！？\n]", text)
    counts: Dict[str, int] = {}
    examples: Dict[str, str] = {}
    for s in sentences:
        norm = normalize_sentence(s)
        if len(norm) < 20:
            continue
        # 法条/文件引用句不参与统计
        if "《" in s and "》" in s:
            continue
        # 方案固有格式行（责任单位/完成时限）在真实公文中
        # 本就逐条重复，不参与统计
        if re.search(r"责任单位|完成时限|牵头单位|配合单位", s):
            continue
        counts[norm] = counts.get(norm, 0) + 1
        examples.setdefault(norm, s.strip()[:60])
    repeated = {
        k: v for k, v in counts.items() if v >= 3
    }
    worst = max(repeated.values()) if repeated else 0
    return {
        "hit": worst >= 4,
        "worst_count": worst,
        "examples": [examples[k] for k in list(repeated)[:5]],
    }


# ---------- 5.4 截断 ----------

DANGLING_ENDINGS = (
    "，", "、", "：", ":", "；", ";", "（", "(",
    "一是", "二是", "三是", "四是", "五是", "六是",
    "第一", "第二", "第三", "第四", "第五",
    "的", "了", "和", "与", "及", "并",
)


def ends_dangling(text: str) -> Tuple[bool, str]:
    tail = text.rstrip()
    if not tail:
        return True, "空输出"
    for ending in DANGLING_ENDINGS:
        if tail.endswith(ending):
            return True, f"以“{ending}”结尾"
    # 书名号未闭合
    if tail.count("《") != tail.count("》"):
        return True, "书名号未闭合"
    # 末尾是孤立的条款编号行
    last_line = tail.split("\n")[-1].strip()
    if re.fullmatch(r"[一二三四五六七八九十]+、?", last_line):
        return True, f"末尾是孤立编号“{last_line}”"
    if re.fullmatch(r"（[一二三四五六七八九十]+）", last_line):
        return True, f"末尾是孤立小标题“{last_line}”"
    return False, ""


def detect_truncation(
    text: str,
    reached_max_tokens: bool,
) -> Dict[str, Any]:
    dangling, reason = ends_dangling(text)
    if reached_max_tokens and dangling:
        return {"hit": True, "reason": f"达到生成长度上限且{reason}"}
    if reached_max_tokens:
        # 达到上限但结尾完整：检查是否有落款收尾
        tail = text[-600:]
        has_date = len(find_dates(tail)) > 0
        if not has_date:
            return {
                "hit": True,
                "reason": "达到生成长度上限且缺少落款收尾",
            }
    return {"hit": False, "reason": ""}


# ---------- 5.5 乱码 ----------

def detect_garbled(text: str) -> Dict[str, Any]:
    hits = []
    if "�" in text:
        hits.append("替换符 �")
    pua = re.findall(r"[\ue000-\uf8ff]", text)
    if pua:
        hits.append(f"私用区字符 ×{len(pua)}")
    if re.search(r"□{3,}", text):
        hits.append("连续 □")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
        hits.append("控制字符")
    return {"hit": len(hits) > 0, "kinds": hits}


# ============================================================
# 6. 检测器 —— 完整性（决定 A / B）
# ============================================================

# 日期识别：阿拉伯数字（允许空格/全角）+ 中文数字
CN_NUM = "零〇一二三四五六七八九十百千"
DATE_PATTERNS = [
    r"20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日",
    r"[二〇零○一二三四五六七八九]{4}\s*年\s*"
    r"[正一二三四五六七八九十]{1,3}\s*月\s*"
    r"[一二三四五六七八九十廿卅]{1,4}\s*日",
]


def normalize_width(text: str) -> str:
    """全角数字/字母转半角，用于日期识别。"""
    return unicodedata.normalize("NFKC", text)


def find_dates(text: str) -> List[str]:
    normalized = normalize_width(text)
    dates = []
    for pattern in DATE_PATTERNS:
        dates.extend(re.findall(pattern, normalized))
    return list(dict.fromkeys(dates))


CN_DIGIT = {
    "零": 0, "〇": 0, "○": 0, "一": 1, "二": 2, "三": 3,
    "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def parse_date_year(date_str: str) -> Optional[int]:
    m = re.search(r"(20\d{2})\s*年", date_str)
    if m:
        return int(m.group(1))
    m = re.search(
        r"([二〇零○一二三四五六七八九]{4})\s*年", date_str
    )
    if m:
        return int("".join(str(CN_DIGIT[c]) for c in m.group(1)))
    return None


def check_structure(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    issues = []

    # 标题：首个非空行
    has_title = False
    if lines:
        first = lines[0]
        if 4 <= len(first) <= 60 and not first.endswith("。"):
            has_title = True
    if not has_title:
        issues.append("缺少规范标题行")

    # 主送机关：前 800 字内以冒号结尾的行
    head = "\n".join(lines)[:800]
    has_recipient = any(
        l.endswith("：") and 2 <= len(l) <= 60
        for l in head.split("\n")
    )
    if not has_recipient:
        issues.append("缺少主送机关")

    # 落款：全文范围搜索单位名行。
    # 印发通知类公文的落款在文中部（通知部分末尾），
    # 所附方案正文之后不再重复落款，因此不能只看尾部。
    has_signoff = any(
        re.search(r"(政府|办公厅|办公室|委员会|局|厅|部|署)$", l)
        and len(l) <= 30
        for l in lines
    )
    if not has_signoff:
        issues.append("缺少落款单位")

    # 成文日期：全文范围搜索具体日期
    has_date = len(find_dates(text)) > 0
    if not has_date:
        issues.append("缺少成文日期")

    return {
        "has_title": has_title,
        "has_recipient": has_recipient,
        "has_signoff": has_signoff,
        "has_date": has_date,
        "issues": issues,
    }


# ---------- 6.1 文号格式校验 ----------

# 宽松抓取各种"长得像文号"的片段，再逐项校验
DOCNUM_CANDIDATE_RE = re.compile(
    r"[\u4e00-\u9fa5]{1,10}"
    r"[〔\[【［]"
    r"[^〕\]】］\n]{1,15}"
    r"[〕\]】］\]]?"
    r"\s*[0-9０-９OoＯｏXx×_\s]{0,8}\s*[号好]"
)


def check_document_numbers(text: str) -> Dict[str, Any]:
    """
    校验文号格式。规范文号：机关代字〔4位年份〕序号+号，
    全角〔〕，阿拉伯数字，无字母，不换行。
    引用历史文件的真实文号同样应符合格式，
    格式异常的一律提示人工修正（B 级）。
    """
    problems = []
    for m in DOCNUM_CANDIDATE_RE.finditer(text):
        raw = m.group(0)
        reasons = []
        normalized = normalize_width(raw)

        # 字母 O/o 冒充数字 0
        if re.search(r"[OoＯｏ]", raw):
            reasons.append("含字母 O（应为数字 0）")
        # 全角数字
        if re.search(r"[０-９]", raw):
            reasons.append("含全角数字")
        # 括号不规范
        if "〔" not in raw or "〕" not in raw:
            reasons.append("括号不是全角〔〕")
        # 结尾字
        if raw.rstrip().endswith("好"):
            reasons.append("“好”应为“号”")
        # 年份位数与合理性
        ym = re.search(r"〔(\d+)", normalized)
        if ym:
            year = ym.group(1)
            if len(year) != 4:
                reasons.append(f"年份 {year} 不是 4 位")
            elif not (1949 <= int(year) <= TODAY.year + 1):
                reasons.append(f"年份 {year} 不合理")
        # 序号位含非数字
        tail = normalized.split("〕")[-1] if "〕" in normalized \
            else normalized
        tail_clean = re.sub(r"[号好\s]", "", tail)
        if tail_clean and not tail_clean.isdigit():
            reasons.append("序号含非数字字符")

        if reasons:
            problems.append({
                "text": raw.replace("\n", " ")[:40],
                "reasons": reasons,
            })

    # 文号被换行拆散：〔 与 〕 之间存在换行
    for m in re.finditer(r"〔[^〕]{0,20}〔?", text):
        seg = m.group(0)
        if "\n" in seg and re.search(r"〔\s*\d", seg):
            problems.append({
                "text": seg.replace("\n", "⏎")[:40],
                "reasons": ["文号被换行拆散"],
            })

    # 去重
    seen = set()
    unique = []
    for p in problems:
        key = p["text"]
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return {"problems": unique[:8], "count": len(unique)}


# ---------- 6.1b 文号自动规范化 ----------

# 训练数据来自政府网站转载/PDF 转换，文号常带：
# 全角数字（〔２０２６〕）、字母 O 冒充 0（〔2O26〕）、
# 错误括号（〔2026] / ［2018］）、序号空格/字母 l（1 2号 / 1l 号）、
# 换行拆散（〔2\n026\n〕\n12\n号）。
# 这些都是机械可修的，不需要人工，也不需要模型重训就能修复。
DOCNUM_LOOSE_RE = re.compile(
    r"([\u4e00-\u9fa5]{1,10})"       # 机关代字
    r"[〔\[【［]"
    r"([0-9０-９OoＯｏIl\s]{1,12})"  # 年份部分
    r"[〕\]】］\]]"
    r"\s*([0-9０-９OoＯｏIl\s]{1,8})"  # 序号部分
    r"\s*[号好]"
)


def _clean_num_part(s: str, letter_map: Dict[str, str]) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", "", s)
    for src, dst in letter_map.items():
        s = s.replace(src, dst)
    return s


def normalize_document_numbers(
    text: str,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    自动修复机械性文号畸形。
    年份位数/合理性问题（2106、212）不猜测，保留原文，
    仍由 check_document_numbers 报 B 级提示。
    返回 (修复后文本, [(原文, 新文), ...])。
    """
    fixes: List[Tuple[str, str]] = []

    def repl(m: "re.Match") -> str:
        prefix, year_raw, num_raw = m.groups()
        year = _clean_num_part(
            year_raw, {"O": "0", "o": "0", "l": "1", "I": "1"}
        )
        num = _clean_num_part(
            num_raw, {"O": "0", "o": "0", "l": "1", "I": "1"}
        )
        if not year.isdigit() or not num.isdigit():
            return m.group(0)
        if not (len(year) == 4 and 1949 <= int(year) <= TODAY.year + 1):
            return m.group(0)  # 年份不合理，不猜
        new = f"{prefix}〔{year}〕{num}号"
        if new != m.group(0):
            fixes.append((m.group(0), new))
        return new

    out = DOCNUM_LOOSE_RE.sub(repl, text)
    return out, fixes


# ---------- 6.1c 落款日期自动校正 ----------

# 模型偶尔复述训练集中历史文档的落款日期（2013/2014 等）。
# 落款日期位置固定（文末 200 字内的最后一个具体日期）、
# 语义明确（本文件成文日期），程序直接替换为今天。
# 正文中的历史引用日期不在文末，不受影响。
def fix_signoff_date(text: str) -> Tuple[str, Optional[str]]:
    normalized = normalize_width(text)
    matches = []
    for pattern in DATE_PATTERNS:
        matches.extend(re.finditer(pattern, normalized))
    if not matches:
        return text, None
    last = matches[-1]
    # 只处理文末 200 字内的日期（即落款日期）
    if last.start() < len(normalized) - 200:
        return text, None
    year = parse_date_year(last.group(0))
    if year is None or (TODAY.year - 1 <= year <= TODAY.year + 1):
        return text, None
    old = last.group(0)
    # NFKC 不改变中文文本长度，位置可直接用于原文
    new_text = text[:last.start()] + TODAY_STR + text[last.end():]
    return new_text, f"落款日期校正：{old} → {TODAY_STR}"


# ---------- 6.2 程序生成当前文号 ----------

# 当前文件文号由程序生成，LLM 禁止自由生成。
# 代字根据落款单位推导；序号由程序按顺序分配，
# 实际部署时由业务系统的发文登记号替换。
DOCNUM_DAIZI_MAP = [
    ("国务院办公厅", "国办发"),
    ("国务院", "国发"),
    ("广东省人民政府办公厅", "粤府办"),
    ("广东省人民政府", "粤府"),
    ("人民政府办公厅", "政办发"),
    ("人民政府", "政发"),
    ("司法局", "司发"),
    ("办公厅", "政办发"),
    ("局", "局发"),
]
DEFAULT_DAIZI = "粤府办"

SIGNOFF_LINE_RE = re.compile(
    r"^[\u4e00-\u9fa5]{2,20}"
    r"(政府|办公厅|办公室|委员会|局|厅|部|署)$"
)

# 文头区域（前 6 个非空行）中"独立成行的文号"
HEADER_DOCNUM_LINE_RE = re.compile(
    r"^[\u4e00-\u9fa5]{0,10}\s*"
    r"[〔\[【［][^〕\]】］\]\n]{1,15}[〕\]】］\]]?"
    r"\s*[0-9０-９OoＯｏIl\s]{0,8}\s*[号好]?$"
)


def derive_daizi(text: str) -> str:
    """从落款单位推导文号代字。"""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in reversed(lines):
        if SIGNOFF_LINE_RE.match(line):
            for key, daizi in DOCNUM_DAIZI_MAP:
                if key in line:
                    return daizi
            break
    return DEFAULT_DAIZI


def strip_header_docnum(text: str) -> Tuple[str, Optional[str]]:
    """
    移除模型在文头自写的独立文号行。
    只处理前 6 个非空行，正文中引用的历史文号不受影响。
    """
    lines = text.split("\n")
    nonempty_idx = [i for i, l in enumerate(lines) if l.strip()]
    stripped = None
    for i in nonempty_idx[:6]:
        line = lines[i].strip()
        if HEADER_DOCNUM_LINE_RE.match(line) and (
            "〔" in line or "［" in line or "[" in line or "【" in line
        ):
            stripped = line
            lines[i] = ""
            break
    return "\n".join(lines), stripped


def inject_program_docnum(text: str, seq: int) -> Tuple[str, str]:
    """在标题行之后插入程序生成的文号行。"""
    daizi = derive_daizi(text)
    docnum = f"{daizi}〔{TODAY.year}〕{seq}号"
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip():
            lines.insert(i + 1, docnum)
            return "\n".join(lines), docnum
    return docnum + "\n" + text, docnum


# ---------- 6.3 生成期退化/超长早停 ----------

class DegenerationEarlyStop:
    """
    生成期早停（transformers StoppingCriteria 协议）：
    - 连续重复片段出现 → 立即停
    - 尾部 4-gram 多样性 < 0.28 → 退化，立即停
    - 生成超过 LONG_GEN_TOKENS 且多样性 < 0.50 → 异常超长，提前停
    健康公文 4-gram 多样性 >= 0.75，两个阈值都留有充分余量。
    """

    def __init__(self, tokenizer, prompt_len: int):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.last_check = 0
        self.stop_reason: Optional[str] = None

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        n = input_ids.shape[-1] - self.prompt_len
        if n - self.last_check < EARLY_STOP_CHECK_EVERY:
            return False
        self.last_check = n

        tail = self.tokenizer.decode(
            input_ids[0, -600:], skip_special_tokens=True
        )
        plain = re.sub(r"\s+", "", tail)
        if len(plain) < 200:
            return False

        m = re.search(r"(.{8,60}?)\1{2,}", plain)
        if m:
            self.stop_reason = f"连续重复片段（第 {n} token 处）"
            return True

        grams = [plain[i:i + 4] for i in range(len(plain) - 3)]
        diversity = len(set(grams)) / len(grams)
        if diversity < 0.28:
            self.stop_reason = (
                f"尾部多样性崩塌 {diversity:.2f}（第 {n} token 处）"
            )
            return True
        if n > LONG_GEN_TOKENS and diversity < 0.50:
            self.stop_reason = (
                f"异常超长且多样性偏低 {diversity:.2f}"
                f"（第 {n} token 处）"
            )
            return True
        return False


# ---------- 6.2 转载/节选标记 ----------

TRANSFER_MARK_PATTERNS = [
    r"（?\(?本文有删减）?\)?",
    r"（?\(?本文有删节）?\)?",
    r"（?\(?节选）?\)?$",
    r"（?\(?下略）?\)?",
    r"（?\(?略）?\)?\s*$",
]


def check_transfer_marks(text: str) -> Dict[str, Any]:
    hits = []
    for pattern in TRANSFER_MARK_PATTERNS:
        hits.extend(re.findall(pattern, text, flags=re.MULTILINE))
    hits = [h for h in dict.fromkeys(hits) if h.strip("（）()")]
    return {"hit": len(hits) > 0, "matches": hits[:5]}


# ---------- 6.3 成文日期合理性 ----------

def check_date_currency(text: str) -> Dict[str, Any]:
    """
    取全文最后一个具体日期视为成文日期。
    年份明显偏离当前年份（草稿应写当天）→ 提示。
    历史引用日期不受影响（只查最后一个）。
    """
    dates = find_dates(text)
    if not dates:
        return {"hit": False}
    last = dates[-1]
    year = parse_date_year(last)
    if year is None:
        return {"hit": False}
    if not (TODAY.year - 1 <= year <= TODAY.year + 1):
        return {
            "hit": True,
            "date": last,
            "reason": f"成文日期“{last}”年份偏离当前年份"
                      f"（{TODAY.year}），使用前需替换",
        }
    return {"hit": False}


def check_relevance(text: str, category: str) -> Dict[str, Any]:
    keywords = RELEVANCE_KEYWORDS.get(category, [category])
    hits = sum(text.count(k) for k in keywords)
    return {
        "ok": hits >= 3,
        "hits": hits,
        "keywords": keywords,
    }


# ============================================================
# 7. 综合评级
# ============================================================

MIN_BODY_CHARS = 800
PASS_A_RATIO = 0.85


def evaluate(
    text: str,
    category: str,
    reached_max_tokens: bool,
) -> Dict[str, Any]:
    critical: List[str] = []
    review: List[str] = []
    detail: Dict[str, Any] = {}

    ph = detect_placeholders(text)
    detail["placeholders"] = ph
    if ph["hit"]:
        critical.append(f"占位符：{ph['matches'][:3]}")

    dg = detect_degeneration(text)
    detail["degeneration"] = dg
    if dg["hit"]:
        critical.append(f"退化循环（{dg['kind']}）")

    rp = detect_repetition(text)
    detail["repetition"] = rp
    if rp["hit"]:
        critical.append(
            f"机械重复：同一句最多出现 {rp['worst_count']} 次"
        )
    elif rp["worst_count"] >= 3:
        review.append(
            f"句子重复 {rp['worst_count']} 次，建议人工检查"
        )

    tr = detect_truncation(text, reached_max_tokens)
    detail["truncation"] = tr
    if tr["hit"]:
        critical.append(f"截断：{tr['reason']}")

    gb = detect_garbled(text)
    detail["garbled"] = gb
    if gb["hit"]:
        critical.append(f"乱码：{gb['kinds']}")

    st = check_structure(text)
    detail["structure"] = st
    review.extend(st["issues"])

    dn = check_document_numbers(text)
    detail["document_numbers"] = dn
    for p in dn["problems"]:
        review.append(
            f"文号格式异常：{p['text']}"
            f"（{'；'.join(p['reasons'])}）"
        )

    tm = check_transfer_marks(text)
    detail["transfer_marks"] = tm
    if tm["hit"]:
        review.append(
            f"含转载/节选标记：{tm['matches']}"
        )

    dc = check_date_currency(text)
    detail["date_currency"] = dc
    if dc["hit"]:
        review.append(dc["reason"])

    rel = check_relevance(text, category)
    detail["relevance"] = rel
    if not rel["ok"]:
        review.append(f"主题相关性弱（关键词命中 {rel['hits']} 次）")

    char_count = len(re.sub(r"\s+", "", text))
    detail["char_count"] = char_count
    if char_count < MIN_BODY_CHARS:
        review.append(f"正文偏短（{char_count} 字 < {MIN_BODY_CHARS}）")

    dangling, reason = ends_dangling(text)
    if dangling and not tr["hit"]:
        review.append(f"结尾悬空（{reason}），未达到长度上限")

    if critical:
        grade = "D"
    elif review:
        grade = "B"
    else:
        grade = "A"

    return {
        "grade": grade,
        "usable": grade == "A",
        "critical_issues": critical,
        "review_issues": review,
        "char_count": char_count,
        "detail": detail,
    }


# ============================================================
# 8. 文本清理与生成
# ============================================================

def clean_generation(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:text|markdown)?\s*", "", text,
                  flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^\s*(assistant|助手)\s*[:：]\s*", "", text,
                  flags=re.IGNORECASE)
    return text.strip()


def load_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quantization,
        device_map="auto",
        trust_remote_code=True,
    )
    if model_name != "base":
        from peft import PeftModel
        adapter = ADAPTERS[model_name]
        print(f"加载 LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def generate(
    model,
    tokenizer,
    test: Dict[str, Any],
    sample: bool = False,
    rep_penalty: float = REPETITION_PENALTY,
) -> Tuple[str, bool, float, Optional[str]]:
    import torch
    from transformers import StoppingCriteria

    messages = [
        {"role": "system",
         "content": SYSTEM_PROMPT.replace("{TODAY}", TODAY_STR)},
        {"role": "user", "content": test["prompt"]},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # MRO 顺序保证 DegenerationEarlyStop.__call__ 生效
    class _EarlyStop(DegenerationEarlyStop, StoppingCriteria):
        def __init__(self):
            DegenerationEarlyStop.__init__(
                self, tokenizer, inputs.shape[-1]
            )

    early_stop = _EarlyStop()

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=rep_penalty,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=[early_stop],
    )
    if sample:
        # 重试时换采样解码，跳出贪心解码的局部陷阱
        gen_kwargs.update(
            do_sample=True, temperature=0.8, top_p=0.95
        )
    else:
        gen_kwargs.update(do_sample=DO_SAMPLE)

    start = time.time()
    with torch.inference_mode():
        output = model.generate(inputs, **gen_kwargs)
    seconds = time.time() - start

    new_tokens = output[0][inputs.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    reached_max = len(new_tokens) >= MAX_NEW_TOKENS
    return (
        clean_generation(text),
        reached_max,
        seconds,
        early_stop.stop_reason,
    )


# ============================================================
# 9. 报告
# ============================================================

GRADE_LABEL = {
    "A": "可直接使用",
    "B": "需少量人工完善",
    "D": "不可用，需重新生成",
}


def build_report(
    model_name: str,
    rows: List[Dict[str, Any]],
) -> str:
    total = len(rows)
    grades = {"A": 0, "B": 0, "D": 0}
    grades_fixed = {"A": 0, "B": 0, "D": 0}
    placeholder_n = degeneration_n = truncation_n = 0
    repetition_n = garbled_n = 0
    early_stop_n = retry_n = retry_adopt_n = 0
    docnum_inject_n = docnum_strip_n = 0
    chars = []

    for r in rows:
        e = r["evaluation"]
        grades[e["grade"]] += 1
        ef = r.get("evaluation_fixed", e)
        grades_fixed[ef["grade"]] += 1
        chars.append(e["char_count"])
        d = e.get("detail") or {}
        if d.get("placeholders", {}).get("hit"):
            placeholder_n += 1
        if d.get("degeneration", {}).get("hit"):
            degeneration_n += 1
        if d.get("truncation", {}).get("hit"):
            truncation_n += 1
        if d.get("repetition", {}).get("hit"):
            repetition_n += 1
        if d.get("garbled", {}).get("hit"):
            garbled_n += 1
        if r.get("early_stop_reason"):
            early_stop_n += 1
        for ev in r.get("inference_events", []):
            if ev.startswith("生成过短"):
                retry_n += 1
            if ev == "采用重试结果":
                retry_adopt_n += 1
            if ev.startswith("程序注入文号"):
                docnum_inject_n += 1
            if ev.startswith("移除模型自写文号"):
                docnum_strip_n += 1

    a_ratio = grades_fixed["A"] / total if total else 0.0
    passed = a_ratio >= PASS_A_RATIO and grades_fixed["D"] == 0

    lines = []
    lines.append("=" * 80)
    lines.append("政府公文生成能力验收报告 V1.2")
    lines.append("=" * 80)
    lines.append("")
    lines.append("验收目标：生成完整、规范、可直接使用的公文。")
    lines.append("评级标准：")
    lines.append("  A = 结构完整、无硬伤，可直接使用")
    lines.append("  B = 需少量人工完善（缺落款/日期、偏短等）")
    lines.append("  D = 有硬伤（占位符/退化循环/截断/乱码/机械重复），需重新生成")
    lines.append(f"通过标准：A 级比例 >= {PASS_A_RATIO:.0%} 且 D 级为 0")
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"模型：{model_name}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"测试数量：{total}")
    lines.append(
        f"原始评级（模型能力）："
        f"A {grades['A']} / B {grades['B']} / D {grades['D']}"
    )
    lines.append(
        f"交付评级（推理层修复后）："
        f"A {grades_fixed['A']}（{a_ratio:.0%}）"
        f" / B {grades_fixed['B']} / D {grades_fixed['D']}"
    )
    lines.append("")
    lines.append(
        f"推理层动作：退化早停 {early_stop_n}，"
        f"短文重试 {retry_n}（采用 {retry_adopt_n}），"
        f"程序注入文号 {docnum_inject_n}，"
        f"移除模型自写文号 {docnum_strip_n}"
    )
    lines.append("")
    lines.append(f"占位符：{placeholder_n}")
    lines.append(f"退化循环：{degeneration_n}")
    lines.append(f"机械重复：{repetition_n}")
    lines.append(f"截断：{truncation_n}")
    lines.append(f"乱码：{garbled_n}")
    if chars:
        lines.append(
            f"平均字数：{sum(chars) / len(chars):.0f}"
            f"（最短 {min(chars)}，最长 {max(chars)}）"
        )
    lines.append("")
    lines.append(
        f"最终结论：{'PASS' if passed else 'FAIL'}"
    )
    lines.append("")
    lines.append("-" * 80)
    lines.append("逐题明细")
    lines.append("-" * 80)

    for r in rows:
        e = r["evaluation"]
        ef = r.get("evaluation_fixed", e)
        grade_str = e["grade"]
        if ef["grade"] != e["grade"]:
            grade_str = f"{e['grade']}→{ef['grade']}"
        lines.append("")
        lines.append(
            f"[{r['id']}] {r['category']} / {r['genre']}"
            f" —— {grade_str}（{GRADE_LABEL[ef['grade']]}）"
            f" {ef['char_count']} 字"
        )
        for ev in r.get("inference_events", []):
            lines.append(f"    推理层：{ev}")
        for issue in ef["critical_issues"]:
            lines.append(f"    硬伤：{issue}")
        for issue in ef["review_issues"]:
            lines.append(f"    提示：{issue}")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# 10. 主流程
# ============================================================

def grade_rank(grade: str) -> int:
    return {"A": 2, "B": 1, "D": 0}.get(grade, 0)


def postprocess(
    text: str, seq: int
) -> Tuple[str, List[str]]:
    """
    推理层后处理流水线：
    1. 剥离模型在文头自写的文号（LLM 禁止自由生成文号）
    2. 规范化正文引用的历史文号（O/０/１/［］ 等格式污染）
    3. 注入程序生成的当前文号
    """
    events: List[str] = []

    text, stripped = strip_header_docnum(text)
    if stripped:
        events.append(f"移除模型自写文号：{stripped[:40]}")

    text, fixes = normalize_document_numbers(text)
    if fixes:
        events.append(f"历史文号格式规范化 {len(fixes)} 处")

    text, date_fix = fix_signoff_date(text)
    if date_fix:
        events.append(date_fix)

    text, docnum = inject_program_docnum(text, seq)
    events.append(f"程序注入文号：{docnum}")

    return text, events


def evaluate_with_events(
    text: str,
    category: str,
    reached_max: bool,
    stop_reason: Optional[str],
) -> Dict[str, Any]:
    e = evaluate(text, category, reached_max)
    if stop_reason:
        e["critical_issues"] = [
            f"生成退化，已提前停止（{stop_reason}）"
        ] + e["critical_issues"]
        e["grade"] = "D"
        e["usable"] = False
    return e


def run_generation(model_name: str) -> List[Dict[str, Any]]:
    tokenizer = load_tokenizer()
    model = load_model(model_name)
    rows = []
    try:
        for seq, test in enumerate(TESTS, start=1):
            print(f"\n===== {test['id']} {test['category']} =====")
            try:
                # 第一次生成（贪心）
                text, reached_max, seconds, stop_reason = generate(
                    model, tokenizer, test
                )
                events: List[str] = []
                if stop_reason:
                    events.append(f"早停：{stop_reason}")

                # 原始评级：模型能力
                eval_raw = evaluate_with_events(
                    text, test["category"], reached_max, stop_reason
                )

                # 后处理 + 交付评级
                final_text, pp_events = postprocess(text, seq)
                events.extend(pp_events)
                eval_final = evaluate_with_events(
                    final_text, test["category"],
                    reached_max, stop_reason,
                )

                # 异常短生成 / 退化早停：标记并重试一次。
                # 退化重试用采样解码 + 更高重复惩罚，
                # 跳出贪心解码的重复陷阱。
                need_retry = False
                if (
                    eval_final["char_count"] < SHORT_RETRY_MIN_CHARS
                    and not stop_reason
                ):
                    events.append(
                        f"生成过短"
                        f"（{eval_final['char_count']} 字"
                        f" < {SHORT_RETRY_MIN_CHARS}），重试一次"
                    )
                    need_retry = True
                    retry_sample = True
                    retry_penalty = REPETITION_PENALTY
                elif stop_reason:
                    events.append("退化早停，重试一次（采样解码）")
                    need_retry = True
                    retry_sample = True
                    retry_penalty = 1.15

                if need_retry:
                    t2, rm2, s2, sr2 = generate(
                        model, tokenizer, test,
                        sample=retry_sample,
                        rep_penalty=retry_penalty,
                    )
                    if sr2:
                        events.append(f"重试早停：{sr2}")
                    e2_raw = evaluate_with_events(
                        t2, test["category"], rm2, sr2
                    )
                    f2, pp2 = postprocess(t2, seq)
                    e2_final = evaluate_with_events(
                        f2, test["category"], rm2, sr2
                    )
                    seconds += s2
                    # 取更优者：先比交付评级，再比字数
                    better = (
                        grade_rank(e2_final["grade"]),
                        e2_final["char_count"],
                    ) > (
                        grade_rank(eval_final["grade"]),
                        eval_final["char_count"],
                    )
                    if better:
                        events.append("采用重试结果")
                        text, final_text = t2, f2
                        eval_raw, eval_final = e2_raw, e2_final
                        reached_max, stop_reason = rm2, sr2
                        events.extend(
                            f"重试后{ev}" for ev in pp2
                        )
                    else:
                        events.append("重试结果未更优，保留首次")

                row = {
                    "model": model_name,
                    **test,
                    "generation": final_text,
                    "generation_raw": text,
                    "inference_events": events,
                    "docnum_fixes": [],
                    "reached_max_tokens": reached_max,
                    "early_stop_reason": stop_reason,
                    "evaluation": eval_raw,
                    "evaluation_fixed": eval_final,
                    "generation_seconds": round(seconds, 2),
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                err_eval = {
                    "grade": "D",
                    "usable": False,
                    "critical_issues": [f"生成异常：{exc}"],
                    "review_issues": [],
                    "char_count": 0,
                    "detail": {},
                }
                row = {
                    "model": model_name,
                    **test,
                    "generation": "",
                    "generation_raw": "",
                    "docnum_fixes": [],
                    "reached_max_tokens": False,
                    "evaluation": err_eval,
                    "evaluation_fixed": err_eval,
                    "generation_seconds": 0,
                    "error": str(exc),
                }
            rows.append(row)
            e = row["evaluation"]
            ef = row["evaluation_fixed"]
            grade_str = e["grade"]
            if ef["grade"] != e["grade"]:
                grade_str = f"{e['grade']}→{ef['grade']}"
            print(
                f"  -> {grade_str} 级，"
                f"{ef['char_count']} 字，"
                f"{row['generation_seconds']}s"
            )
            for ev in row["inference_events"]:
                print(f"     [推理层] {ev}")
            # 每题保存纯净正文，A 级可直接使用
            article_file = (
                OUT_DIR / f"{test['id']}_{test['category']}.txt"
            )
            article_file.write_text(
                row["generation"], encoding="utf-8"
            )
    finally:
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return rows


def score_only(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for seq, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        old = json.loads(line)
        text = old.get("generation_raw") or old.get("generation", "")
        reached_max = old.get("reached_max_tokens", False)
        stop_reason = old.get("early_stop_reason")
        evaluation = evaluate_with_events(
            text, old.get("category", ""), reached_max, stop_reason
        )
        fixed_text, events = postprocess(text, seq)
        evaluation_fixed = evaluate_with_events(
            fixed_text, old.get("category", ""),
            reached_max, stop_reason,
        )
        rows.append({
            "model": old.get("model", "unknown"),
            "id": old.get("id"),
            "category": old.get("category"),
            "genre": old.get("genre"),
            "prompt": old.get("prompt"),
            "generation": fixed_text,
            "generation_raw": text,
            "inference_events": events,
            "docnum_fixes": [],
            "reached_max_tokens": reached_max,
            "early_stop_reason": stop_reason,
            "evaluation": evaluation,
            "evaluation_fixed": evaluation_fixed,
            "generation_seconds": old.get("generation_seconds"),
            "error": old.get("error"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["base", "v1", "v11", "v12", "all"],
        default="v12",
    )
    parser.add_argument(
        "--score-only",
        type=str,
        default=None,
        help="对已有 generation_results.jsonl 重新评分，"
             "不加载模型",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        rows = score_only(Path(args.score_only))
        model_name = rows[0]["model"] if rows else "unknown"
        # 同步导出修复后的纯净正文
        for row in rows:
            if row.get("generation"):
                article_file = (
                    OUT_DIR
                    / f"{row['id']}_{row['category']}.txt"
                )
                article_file.write_text(
                    row["generation"], encoding="utf-8"
                )
    else:
        model_names = (
            ["base", "v1", "v11", "v12"]
            if args.model == "all"
            else [args.model]
        )
        rows = []
        model_name = args.model
        for name in model_names:
            rows.extend(run_generation(name))

    result_file = OUT_DIR / "generation_results_v12.jsonl"
    with open(result_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    report = build_report(model_name, rows)
    report_file = OUT_DIR / "generation_acceptance_report_v12.txt"
    report_file.write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\n结果：{result_file}")
    print(f"报告：{report_file}")
    print(f"正文：{OUT_DIR}/H*.txt")


if __name__ == "__main__":
    main()