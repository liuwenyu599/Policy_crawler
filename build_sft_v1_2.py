#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_sft_v1_2.py — 把 V1.2 清洗后的文档转换成训练用 messages 格式

输入：v1_2/train.jsonl、v1_2/val.jsonl、v1_2/test.jsonl（clean_v1_2.py 产物）
输出：v1_2/sft/train_real_v12.jsonl / val_real_v12.jsonl / test_real_v12.jsonl

要点：
  * 直接使用清洗后的真实公文原文（不注入占位符、不改写、不扩写）
  * instruction 模板与 V1 完全一致（3 个通用模板轮换 + 1/4 带标题），保证可比性
  * 不生成 synthetic_dialogue（V1 训练只用 real，保持一致）
"""

import json
from pathlib import Path

INSTRUCTIONS = [
    "请撰写一篇符合广东省政策公文规范的正式公文。",
    "请起草一份正式的广东省政策公文。",
    "请根据广东省党政机关公文写作规范撰写一篇政策公文。",
]
INSTRUCTION_WITH_TITLE = "请起草一份《{title}》，要求符合广东省政策公文规范。"


def build(doc: dict, idx: int) -> dict:
    title = (doc.get("title") or "").strip()
    if idx % 4 == 3 and title:
        instruction = INSTRUCTION_WITH_TITLE.format(title=title)
    else:
        instruction = INSTRUCTIONS[idx % len(INSTRUCTIONS)]
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": doc["text"]},
        ],
        "metadata": {"source_UID": doc.get("UID", ""), "data_type": "real_sft"},
    }


def main():
    base = Path("v1_2")
    out = base / "sft"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        src = base / f"{name}.jsonl"
        dst = out / f"{name}_real_v12.jsonl"
        n = 0
        with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
            for i, line in enumerate(fi):
                doc = json.loads(line)
                if not doc.get("text"):
                    continue
                fo.write(json.dumps(build(doc, i), ensure_ascii=False) + "\n")
                n += 1
        print(f"{dst}: {n} 条")


if __name__ == "__main__":
    main()
