#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
Qwen2.5-14B-Instruct
广东省政策公文 Real SFT V1.2 - QLoRA
============================================================

数据：
    train:
        v1_2/sft/train_real_v12.jsonl
        2651 条

    validation:
        v1_2/sft/val_real_v12.jsonl
        147 条

V1：
    只使用真实广东政策公文
    不使用 synthetic dialogue

模型：
    Qwen2.5-14B-Instruct

训练：
    4-bit NF4 QLoRA
    LoRA r=16
    LoRA alpha=32
    LoRA dropout=0.05

长度：
    max_length=8192

测试：
    python train_sft_v1_2.py --test

正式：
    python train_sft_v1_2.py --full

============================================================
"""

import os

# ============================================================
# NCCL
# RTX 4090 当前环境关闭 P2P / IB
# 必须在 import torch 前设置
# ============================================================

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 减少 CUDA allocator 碎片问题
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True"
)

import json
import argparse
from pathlib import Path

import torch
import bitsandbytes as bnb
import torch.distributed as dist
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import LoraConfig

from trl import (
    SFTTrainer,
    SFTConfig,
)


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(
    "/home/lwy/Policy_crawler"
)

MODEL_PATH = Path(
    "/home/lwy/Qwen2.5-14B-Instruct"
)

TRAIN_FILE = (
    PROJECT_ROOT
    / "v1_2/sft/train_real_v12.jsonl"
)

VAL_FILE = (
    PROJECT_ROOT
    / "v1_2/sft/val_real_v12.jsonl"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "v1_2/checkpoints"
)


# ============================================================
# 2. 训练配置
# ============================================================

MAX_SEQ_LENGTH = 8192

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

DEFAULT_EPOCHS = 1.0
DEFAULT_LR = 2e-4


# ============================================================
# 3. 参数
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Qwen2.5-14B 广东政策公文 QLoRA SFT V1.2"
        )
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--test",
        action="store_true",
        help="只使用10条数据进行测试",
    )

    group.add_argument(
        "--full",
        action="store_true",
        help="使用完整2656条数据训练",
    )

    parser.add_argument(
        "--epochs",
        type=float,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    return parser.parse_args()


# ============================================================
# 4. 环境检查
# ============================================================

def check_environment():

    print()
    print("=" * 70)
    print("环境检查")
    print("=" * 70)

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA:",
        torch.version.cuda
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    print(
        "GPU count:",
        torch.cuda.device_count()
    )

    print(
        "WORLD_SIZE:",
        os.environ.get("WORLD_SIZE", "1")
    )

    print(
        "LOCAL_RANK:",
        os.environ.get("LOCAL_RANK", "0")
    )

    print(
        "bitsandbytes:",
        bnb.__version__
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA 不可用。"
        )

    for i in range(
        torch.cuda.device_count()
    ):

        props = torch.cuda.get_device_properties(i)

        total_gb = (
            props.total_memory
            / 1024**3
        )

        print(
            f"GPU {i}: "
            f"{props.name} | "
            f"{total_gb:.2f} GB"
        )

    print()

    print(
        "NCCL_P2P_DISABLE =",
        os.environ.get(
            "NCCL_P2P_DISABLE"
        )
    )

    print(
        "NCCL_IB_DISABLE =",
        os.environ.get(
            "NCCL_IB_DISABLE"
        )
    )

    print(
        "PYTORCH_CUDA_ALLOC_CONF =",
        os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF"
        )
    )

    print()


# ============================================================
# 5. 文件检查
# ============================================================

def check_files():

    print("=" * 70)
    print("文件检查")
    print("=" * 70)

    files = [
        ("模型", MODEL_PATH),
        ("训练集", TRAIN_FILE),
        ("验证集", VAL_FILE),
    ]

    for name, path in files:

        print(
            f"{name}: {path}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"{name}不存在：{path}"
            )

    print()


# ============================================================
# 6. JSONL 检查
# ============================================================

def inspect_jsonl(
    path,
    name,
):

    count = 0
    uids = set()

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1,
        ):

            if not line.strip():
                continue

            obj = json.loads(line)

            if "messages" not in obj:

                raise ValueError(
                    f"{name} 第 {line_no} 条缺少 messages"
                )

            messages = obj["messages"]

            if not isinstance(
                messages,
                list,
            ):

                raise ValueError(
                    f"{name} 第 {line_no} 条 "
                    "messages 不是 list"
                )

            roles = []

            for msg in messages:

                if not isinstance(
                    msg,
                    dict,
                ):

                    raise ValueError(
                        f"{name} 第 {line_no} 条 "
                        "message 不是 dict"
                    )

                role = msg.get("role")
                content = msg.get("content")

                if role is None:
                    raise ValueError(
                        f"{name} 第 {line_no} 条 "
                        "缺少 role"
                    )

                if content is None:
                    raise ValueError(
                        f"{name} 第 {line_no} 条 "
                        "缺少 content"
                    )

                if not isinstance(
                    content,
                    str,
                ):

                    raise ValueError(
                        f"{name} 第 {line_no} 条 "
                        "content 不是字符串"
                    )

                if not content.strip():

                    raise ValueError(
                        f"{name} 第 {line_no} 条 "
                        "content 为空"
                    )

                roles.append(role)

            if "user" not in roles:

                raise ValueError(
                    f"{name} 第 {line_no} 条没有 user"
                )

            if "assistant" not in roles:

                raise ValueError(
                    f"{name} 第 {line_no} 条没有 assistant"
                )

            metadata = obj.get(
                "metadata",
                {}
            )

            if isinstance(
                metadata,
                dict,
            ):

                uid = metadata.get(
                    "source_UID"
                )

                if uid:

                    if uid in uids:

                        raise ValueError(
                            f"{name} 出现重复 UID：{uid}"
                        )

                    uids.add(uid)

            count += 1

    print(
        f"{name}: {count} 条"
    )

    print(
        f"{name}: UID {len(uids)} 个"
    )

    print()

    return count


# ============================================================
# 7. 创建10条测试集
# ============================================================

def create_test_file():

    test_file = (
        PROJECT_ROOT
        / "v1_2/sft/train_real_test10.jsonl"
    )

    lines = []

    with open(
        TRAIN_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            lines.append(line)

            if len(lines) == 10:
                break

    if len(lines) != 10:

        raise RuntimeError(
            "训练集不足10条。"
        )

    with open(
        test_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.writelines(lines)

    print(
        "测试集：",
        test_file
    )

    return test_file


# ============================================================
# 8. Dataset
# ============================================================

def load_data(
    test_mode,
):

    if test_mode:

        train_path = create_test_file()

    else:

        train_path = TRAIN_FILE

    print("=" * 70)
    print("加载 Dataset")
    print("=" * 70)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(VAL_FILE),
        },
    )

    print(
        dataset
    )

    print(
        "train:",
        len(dataset["train"])
    )

    print(
        "validation:",
        len(dataset["validation"])
    )

    print()

    return dataset


# ============================================================
# 9. Tokenizer
# ============================================================

def load_tokenizer():

    print("=" * 70)
    print("加载 Tokenizer")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(
        "pad_token:",
        repr(tokenizer.pad_token)
    )

    print(
        "eos_token:",
        repr(tokenizer.eos_token)
    )

    print()

    return tokenizer


# ============================================================
# 10. 4-bit QLoRA 模型
# ============================================================

def load_model():

    print("=" * 70)
    print("加载 Qwen2.5-14B-Instruct")
    print("模式：4-bit QLoRA + DDP")
    print("=" * 70)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("4-bit quantization: ON")
    print("quant type: NF4")
    print("compute dtype: bfloat16")
    print("double quant: ON")
    print()

    # --------------------------------------------------------
    # 关键修复：
    # 不要在普通 `python train_sft_v1_2.py` 下让 Trainer
    # 自动走 DataParallel。
    #
    # 两张 4090 必须使用 torchrun/DDP：
    #   torchrun --standalone --nproc_per_node=2 \
    #       train_sft_v1_2.py --test
    #
    # 每个 DDP process 只把完整 4-bit 模型放到自己的 GPU。
    # 这样 input_ids、embedding weight、LoRA 参数都在同一设备。
    # --------------------------------------------------------

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank}, "
                f"但当前可见 GPU 数量只有 {torch.cuda.device_count()}。"
            )
        torch.cuda.set_device(local_rank)
        model_device = local_rank

        print(
            f"DDP 模式：WORLD_SIZE={world_size}, "
            f"LOCAL_RANK={local_rank}, "
            f"GPU={torch.cuda.get_device_name(local_rank)}"
        )
    else:
        # 单卡运行保持兼容。
        torch.cuda.set_device(0)
        model_device = 0
        print(
            f"单 GPU 模式：GPU={torch.cuda.get_device_name(0)}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map={"": model_device},
    )

    model.config.use_cache = False

    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False

    # 明确检查：量化模型主参数必须位于当前 process 的 GPU。
    first_param = next(model.parameters())
    actual_device = first_param.device
    expected_device = torch.device(f"cuda:{model_device}")

    print(f"模型首参数设备：{actual_device}")
    print(f"当前进程目标设备：{expected_device}")

    if actual_device != expected_device:
        raise RuntimeError(
            f"模型设备错误：模型在 {actual_device}，"
            f"但当前 process 应在 {expected_device}。"
        )

    print("模型加载成功。")
    print()

    return model


# ============================================================
# 11. LoRA
# ============================================================

def create_lora_config():

    print("=" * 70)
    print("LoRA 配置")
    print("=" * 70)

    print(
        "r =", LORA_R
    )

    print(
        "alpha =", LORA_ALPHA
    )

    print(
        "dropout =", LORA_DROPOUT
    )

    print(
        "targets =",
        ", ".join(TARGET_MODULES)
    )

    config = LoraConfig(

        r=LORA_R,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        bias="none",

        task_type="CAUSAL_LM",

        target_modules=TARGET_MODULES,
    )

    print()

    return config


# ============================================================
# 12. Training Config
# ============================================================

def create_training_config(
    args,
    output_dir,
):

    print("=" * 70)
    print("Training Config")
    print("=" * 70)

    if args.test:

        eval_steps = 5
        save_steps = 5

    else:

        eval_steps = 100
        save_steps = 200

    config = SFTConfig(

        output_dir=str(
            output_dir
        ),

        num_train_epochs=args.epochs,

        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),

        learning_rate=args.lr,

        lr_scheduler_type="cosine",

        warmup_ratio=0.05,

        # ----------------------------------------------------
        # TRL 1.12.0
        # ----------------------------------------------------

        max_length=MAX_SEQ_LENGTH,

        # ----------------------------------------------------
        # BF16
        # ----------------------------------------------------

        bf16=True,

        fp16=False,

        # ----------------------------------------------------
        # Gradient checkpoint
        # ----------------------------------------------------

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optim="paged_adamw_8bit",

        weight_decay=0.01,

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        logging_strategy="steps",

        logging_steps=1,

        logging_first_step=True,

        # ----------------------------------------------------
        # Eval
        # ----------------------------------------------------

        eval_strategy="steps",

        eval_steps=eval_steps,

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_strategy="steps",

        save_steps=save_steps,

        save_total_limit=2,

        # ----------------------------------------------------
        # Misc
        # ----------------------------------------------------

        report_to="none",

        remove_unused_columns=False,

        ddp_find_unused_parameters=False,

        seed=42,

        data_seed=42,

        dataloader_num_workers=2,

        save_safetensors=True,
    )

    print(
        "max_length:",
        MAX_SEQ_LENGTH
    )

    print(
        "batch:",
        PER_DEVICE_TRAIN_BATCH_SIZE
    )

    print(
        "gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS
    )

    print(
        "effective batch per GPU:",
        GRADIENT_ACCUMULATION_STEPS
    )

    print(
        "learning rate:",
        args.lr
    )

    print(
        "epochs:",
        args.epochs
    )

    print(
        "optimizer:",
        "paged_adamw_8bit"
    )

    print()

    return config


# ============================================================
# 13. 显存
# ============================================================

def print_gpu_memory(
    title,
):

    print("=" * 70)
    print(title)
    print("=" * 70)

    for i in range(
        torch.cuda.device_count()
    ):

        allocated = (
            torch.cuda.memory_allocated(i)
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved(i)
            / 1024**3
        )

        print(
            f"GPU {i}: "
            f"allocated={allocated:.2f} GB | "
            f"reserved={reserved:.2f} GB"
        )

    print()


# ============================================================
# 14. Trainer
# ============================================================

def create_trainer(
    model,
    tokenizer,
    dataset,
    lora_config,
    training_config,
):

    print("=" * 70)
    print("创建 SFTTrainer")
    print("=" * 70)

    trainer = SFTTrainer(

        model=model,

        args=training_config,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        processing_class=tokenizer,

        peft_config=lora_config,
    )

    print(
        "SFTTrainer 创建成功。"
    )

    print()

    return trainer


# ============================================================
# 15. 保存配置
# ============================================================

def save_config(
    output_dir,
    args,
    train_count,
    val_count,
):

    config = {

        "model": str(
            MODEL_PATH
        ),

        "train_file": str(
            TRAIN_FILE
        ),

        "val_file": str(
            VAL_FILE
        ),

        "train_count": train_count,

        "val_count": val_count,

        "epochs": args.epochs,

        "learning_rate": args.lr,

        "max_seq_length": MAX_SEQ_LENGTH,

        "quantization": {

            "method": "QLoRA",

            "load_in_4bit": True,

            "quant_type": "nf4",

            "compute_dtype": "bfloat16",

            "double_quant": True,
        },

        "lora": {

            "r": LORA_R,

            "alpha": LORA_ALPHA,

            "dropout": LORA_DROPOUT,

            "target_modules": TARGET_MODULES,
        },

        "environment": {

            "torch": torch.__version__,

            "cuda": torch.version.cuda,

            "gpu_count": (
                torch.cuda.device_count()
            ),

            "bitsandbytes": bnb.__version__,

            "nccl_p2p_disable": (
                os.environ.get(
                    "NCCL_P2P_DISABLE"
                )
            ),

            "nccl_ib_disable": (
                os.environ.get(
                    "NCCL_IB_DISABLE"
                )
            ),
        },
    }

    output_file = (
        output_dir
        / "training_config.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "训练配置：",
        output_file
    )

    print()


# ============================================================
# 16. Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # 多 GPU 必须使用 torchrun/DDP。
    # 旧版本在普通 python 启动时会被 Transformers Trainer
    # 自动包装成 DataParallel；对于 4-bit QLoRA + device_map，
    # replica 1 会出现 embedding weight 在 cuda:0、input_ids
    # 在 cuda:1 的设备冲突。
    # --------------------------------------------------------

    visible_gpu_count = torch.cuda.device_count()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if visible_gpu_count > 1 and world_size == 1:
        raise RuntimeError(
            "\n"
            "检测到多张 GPU，但当前不是 DDP 启动。\n"
            "请不要使用：python train_sft_v1_2.py --full\n"
            "请改用：\n"
            "  torchrun --standalone --nproc_per_node=2 "
            "train_sft_v1_2.py --full\n"
            "测试模式：\n"
            "  torchrun --standalone --nproc_per_node=2 "
            "train_sft_v1_2.py --test\n"
            "这样每张 GPU 各运行一个 DDP process，避免 "
            "DataParallel 的 cuda:0/cuda:1 embedding 冲突。"
        )

    # --------------------------------------------------------
    # 环境
    # --------------------------------------------------------

    check_environment()

    # --------------------------------------------------------
    # 文件
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # 数据检查
    # --------------------------------------------------------

    train_count = inspect_jsonl(
        TRAIN_FILE,
        "train"
    )

    val_count = inspect_jsonl(
        VAL_FILE,
        "validation"
    )

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------

    mode = (
        "test"
        if args.test
        else "full"
    )

    output_dir = (
        OUTPUT_ROOT
        / f"real_sft_v12_lora_{mode}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("广东政策公文 Real SFT V1.2")
    print("QLoRA")
    print("=" * 70)

    print(
        "mode:",
        mode
    )

    print(
        "model:",
        MODEL_PATH
    )

    print(
        "train:",
        train_count
    )

    print(
        "validation:",
        val_count
    )

    print(
        "output:",
        output_dir
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = load_data(
        test_mode=args.test
    )

    actual_train_count = len(
        dataset["train"]
    )

    actual_val_count = len(
        dataset["validation"]
    )

    # --------------------------------------------------------
    # tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # model
    # --------------------------------------------------------

    model = load_model()

    print_gpu_memory(
        "模型加载后"
    )

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    lora_config = (
        create_lora_config()
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    training_config = (
        create_training_config(
            args,
            output_dir,
        )
    )

    # --------------------------------------------------------
    # save config
    # --------------------------------------------------------

    save_config(
        output_dir,
        args,
        actual_train_count,
        actual_val_count,
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = create_trainer(
        model,
        tokenizer,
        dataset,
        lora_config,
        training_config,
    )

    print_gpu_memory(
        "Trainer 创建后"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("=" * 70)
    print("开始训练")
    print("=" * 70)

    print()

    result = trainer.train()

    # --------------------------------------------------------
    # Save adapter
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("保存 LoRA Adapter")
    print("=" * 70)

    trainer.save_model(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    # --------------------------------------------------------
    # train metrics
    # --------------------------------------------------------

    train_metrics = (
        result.metrics
    )

    train_metrics_file = (
        output_dir
        / "train_metrics.json"
    )

    with open(
        train_metrics_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            train_metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "Train metrics:"
    )

    print(
        json.dumps(
            train_metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("最终验证集评估")
    print("=" * 70)

    eval_metrics = (
        trainer.evaluate()
    )

    print(
        json.dumps(
            eval_metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    eval_metrics_file = (
        output_dir
        / "eval_metrics.json"
    )

    with open(
        eval_metrics_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            eval_metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    print_gpu_memory(
        "训练完成"
    )

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    print("=" * 70)
    print("训练完成")
    print("=" * 70)

    print(
        "LoRA Adapter:",
        output_dir
    )

    print(
        "Train metrics:",
        train_metrics_file
    )

    print(
        "Eval metrics:",
        eval_metrics_file
    )
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    print()


if __name__ == "__main__":

    main()
