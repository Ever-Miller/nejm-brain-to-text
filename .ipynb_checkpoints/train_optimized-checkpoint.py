#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整训练脚本（已集成安全的 tokenizer resize 流程）
目标：在显存有限（例如 10GB A100）的情况下，尽量避免 resize_token_embeddings 引发 OOM。
流程简介：
1. 先加载 tokenizer 并 add_tokens（如果有 extra phonemes）
2. 在 CPU 上加载 model 的副本并安全地完成 resize（或手动扩展 embedding），把结果保存到临时目录
3. 根据 use_4bit 选项，从临时目录重新加载模型（4-bit / device_map='auto'），再 move 到训练设备
4. 常规训练流程（LoRA/PEFT、gradient checkpointing、mixed precision 等）
"""
import os
import json
import argparse
import math
import time
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# Optional imports
try:
    import bitsandbytes as bnb  # may reduce memory for some models
    HAS_BNB = True
except Exception:
    HAS_BNB = False

try:
    from peft import get_peft_model, LoraConfig, TaskType
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

# ----------------------------------------------------------------------------
# Dataset (expects header: phonemes\tsentence; phonemes separated by "|")
# ----------------------------------------------------------------------------
class PhonemeTSVDataset(Dataset):
    def __init__(self, tsv_path, phoneme_vocab=None, tokenizer=None, max_length=512):
        self.rows = []
        with open(tsv_path, "r", encoding="utf-8") as fh:
            header = fh.readline()  # skip header
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    ph_str, sent = ln.split("\t", 1)
                except ValueError:
                    # skip malformed lines
                    continue
                phs = [p.strip() for p in ph_str.split("|") if p.strip() and p.strip() != "BLANK"]
                self.rows.append({"phonemes": phs, "sentence": sent})

        # build phoneme vocab if not provided
        if phoneme_vocab is None:
            self.phoneme_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
            idx = max(self.phoneme_vocab.values()) + 1
            for r in self.rows:
                for ph in r["phonemes"]:
                    if ph not in self.phoneme_vocab:
                        self.phoneme_vocab[ph] = idx
                        idx += 1
        else:
            self.phoneme_vocab = phoneme_vocab

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        # join phonemes by space for prompt embedding; we will use tokenizer for whole prompt
        phoneme_str = " ".join(r["phonemes"])
        return {"phoneme_str": phoneme_str, "sentence": r["sentence"]}

# collator: create prompt text and tokenize with HF tokenizer
def collate_batch(batch, tokenizer, max_length=512):
    prompts = []
    for item in batch:
        prompt = (
            "### Instruction:\nConvert the following phoneme sequence into an English sentence.\n\n"
            "### Phonemes:\n" + item["phoneme_str"] + "\n\n### Output:\n"
        )
        full = prompt + item["sentence"]
        prompts.append(full)

    tok = tokenizer(prompts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return tok

# ----------------------------------------------------------------------------
# Utilities: 安全 resize 并保存，确保在 CPU 上进行
# ----------------------------------------------------------------------------

def safe_resize_and_maybe_save(model, tokenizer, tmp_save_dir, mean_resizing=True):
    """
    安全地在 CPU 上完成 resize，然后把模型保存到 tmp_save_dir 以便后续按需用 4bit/auto device_map 重新加载。
    - 假定当前 model 已经在 CPU 上（或能迁移到 CPU）。
    - mean_resizing: 当需要用均值初始化新 token 时设为 True。
    返回：tmp_save_dir 的路径字符串
    """
    tmp_dir = Path(tmp_save_dir)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 强制切到 CPU 以降低 OOM 风险
    try:
        model.to("cpu")
    except Exception:
        pass

    # 清空 cuda cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    old_vocab_size = None
    try:
        old_vocab_size = model.get_input_embeddings().weight.size(0)
    except Exception:
        # Try different attribute names
        try:
            old_vocab_size = model.get_input_embeddings().weight.shape[0]
        except Exception:
            old_vocab_size = None

    new_vocab_size = len(tokenizer)
    print(f"[resize] old_vocab_size={old_vocab_size}, new_vocab_size={new_vocab_size}")

    if old_vocab_size is None:
        print("[resize] Warning: couldn't determine old vocab size, attempting model.resize_token_embeddings directly.")
        try:
            model.resize_token_embeddings(new_vocab_size)
        except Exception as e:
            print("[resize] resize_token_embeddings failed:", e)
    else:
        if new_vocab_size == old_vocab_size:
            print("[resize] token count unchanged, skipping resize.")
        else:
            # 尝试使用 transformers 提供的 API（在 CPU 上）
            tried = False
            try:
                model.resize_token_embeddings(new_vocab_size, mean_resizing=mean_resizing)
                print("[resize] model.resize_token_embeddings succeeded.")
                tried = True
            except TypeError:
                try:
                    model.resize_token_embeddings(new_vocab_size)
                    print("[resize] model.resize_token_embeddings succeeded (no mean_resizing arg).")
                    tried = True
                except Exception as e:
                    print("[resize] resize API failed:", e)
                    tried = False
            except RuntimeError as e:
                print("[resize] resize raised RuntimeError (possible OOM):", e)
                tried = False
            except Exception as e:
                print("[resize] resize raised unexpected exception:", e)
                tried = False

            # 如果 API 没有按预期完成，手动扩展 embedding（CPU 上）
            try:
                emb = model.get_input_embeddings()
                if emb.weight.size(0) != new_vocab_size:
                    print("[resize] Performing manual embedding expansion on CPU.")
                    old_w = emb.weight.detach().cpu()
                    old_n, dim = old_w.size()
                    new_w = torch.zeros((new_vocab_size, dim), dtype=old_w.dtype)
                    new_w[:old_n, :] = old_w
                    if mean_resizing:
                        mean_row = old_w.mean(dim=0, keepdim=True)
                        new_w[old_n:, :] = mean_row.repeat(new_vocab_size - old_n, 1)
                    else:
                        std = old_w.std().item() if old_w.std().item() > 0 else 0.02
                        new_w[old_n:, :] = torch.normal(mean=0.0, std=std, size=(new_vocab_size - old_n, dim))

                    # create and set new embedding module
                    new_emb = torch.nn.Embedding(new_vocab_size, dim)
                    new_emb.weight.data.copy_(new_w)
                    # set into model
                    try:
                        model.set_input_embeddings(new_emb)
                    except Exception:
                        # fallback: try common attr names
                        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                            model.model.embed_tokens = new_emb
                        elif hasattr(model, "embed_tokens"):
                            model.embed_tokens = new_emb
                        else:
                            raise RuntimeError("Cannot set new input embeddings; model structure unexpected.")

                    # update lm_head if present and size mismatch
                    try:
                        if getattr(model, "lm_head", None) is not None:
                            old_lm = model.lm_head.weight.detach().cpu()
                            if old_lm.size(0) != new_vocab_size:
                                out_dim = old_lm.size(1)
                                new_lm = torch.zeros((new_vocab_size, out_dim), dtype=old_lm.dtype)
                                new_lm[:old_lm.size(0), :] = old_lm
                                if mean_resizing:
                                    new_lm[old_lm.size(0):, :] = old_lm.mean(dim=0, keepdim=True).repeat(new_vocab_size - old_lm.size(0), 1)
                                else:
                                    std_lm = old_lm.std().item() if old_lm.std().item() > 0 else 0.02
                                    new_lm[old_lm.size(0):, :] = torch.normal(mean=0.0, std=std_lm, size=(new_vocab_size - old_lm.size(0), out_dim))
                                model.lm_head.weight = torch.nn.Parameter(new_lm)
                    except Exception as e:
                        print("[resize] warning: could not update lm_head automatically:", e)
            except Exception as e:
                print("[resize] manual embedding expansion failed:", e)
                raise

    # Save intermediate model to tmp dir so we can reload it in 4-bit / with device_map
    try:
        model.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        print(f"[resize] Saved resized model+tokenizer to {tmp_dir}")
    except Exception as e:
        print("[resize] Failed to save resized model:", e)
        raise

    # Clear cache to free memory before reload
    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return str(tmp_dir)

# helpful small printer
def print_startup_info(args, tokenizer, model):
    print("\n==== STARTUP INFO ====")
    print(f"Model: {args.model_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Tokenizer vocab size before add: {len(tokenizer)}")
    try:
        print(f"Model config hidden size: {model.config.hidden_size}")
    except Exception:
        pass
    print(f"Per-device batch size: {args.per_device_batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Using LoRA: {args.use_lora and HAS_PEFT}")
    print("======================\n")

# ----------------------------------------------------------------------------
# Training loop (simple but robust)
# ----------------------------------------------------------------------------

def train_loop(args):
    # set a helpful cuda allocator config to reduce fragmentation on some systems
    if torch.cuda.is_available():
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # add phoneme tokens to tokenizer BEFORE model load (减小 GPU resize 风险)
    phonemes_list = args.extra_phonemes.split(",") if args.extra_phonemes else []
    phonemes_list = [p.strip() for p in phonemes_list if p.strip()]
    if phonemes_list:
        added = tokenizer.add_tokens(phonemes_list)
        print(f"Added {added} phoneme tokens to tokenizer (PRE-LOAD)")

    # load model onto CPU first for safe resize
    print("Loading model onto CPU first for safe resize...")
    model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    model_cpu = None
    try:
        # try explicit CPU device_map where supported
        model_cpu = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": "cpu"}, **model_kwargs)
    except Exception:
        # fallback
        model_cpu = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    # safe resize on CPU and save to temporary directory
    tmp_resized_dir = safe_resize_and_maybe_save(model_cpu, tokenizer, tmp_save_dir=os.path.join(args.output_dir, "tmp_resized"), mean_resizing=True)

    # reload model from resized checkpoint with desired dtype/device config
    print("Reloading model from resized checkpoint with desired dtype/device config...")
    model = None
    if args.use_4bit and HAS_BNB:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                tmp_resized_dir,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
            print("Model reloaded in 4-bit (bnb) with device_map='auto'")
        except Exception as e:
            print("Failed to reload in 4-bit, error:", e)
            print("Falling back to fp16/low_cpu_mem_usage load.")
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(tmp_resized_dir, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True)
            print(f"Model reloaded with dtype={dtype} and device_map='auto' (fallback)")
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(tmp_resized_dir, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True)
            print(f"Model reloaded with dtype={dtype} and device_map='auto'")
        except Exception as e:
            print("device_map='auto' load failed, trying without device_map.")
            model = AutoModelForCausalLM.from_pretrained(tmp_resized_dir, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
            print("Model reloaded without device_map; will move to device manually.")

    # print info
    print_startup_info(args, tokenizer, model)

    # apply LoRA if requested and peft available
    if args.use_lora:
        if not HAS_PEFT:
            raise RuntimeError("PEFT is not installed but --use_lora was requested. Install peft to use LoRA.")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
            modules_to_save=["embed_tokens", "lm_head"]
        )
        model = get_peft_model(model, peft_config)
        print("Applied LoRA adapters to model")

        # 确保只有 LoRA 参数（及 embed/lm_head）需要训练
        for name, param in model.named_parameters():
            if 'lora' in name or ('embed_tokens' in name) or ('lm_head' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        print("Verified LoRA parameter freezing (including resized embeddings).")

        # 打印可训练参数量（PEFT 提供的方法）
        try:
            model.print_trainable_parameters()
        except Exception:
            # fallback: count manually
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Trainable params: {trainable} / {total}")

    # enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("Enabled model.gradient_checkpointing")
        except Exception:
            print("Model does not support gradient_checkpointing_enable()")

    # FINALLY move model to CUDA/GPU after all memory-intensive pre-processing
    try:
        model.to(device)
    except Exception:
        print("Warning: model.to(device) failed; continuing (weights may be sharded by device_map).")

    # load datasets
    train_ds = PhonemeTSVDataset(args.train_tsv)
    val_ds = PhonemeTSVDataset(args.val_tsv, phoneme_vocab=train_ds.phoneme_vocab) if args.val_tsv else None

    # save phoneme vocab for inference
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "phoneme_vocab.json"), "w", encoding="utf-8") as fh:
        json.dump(train_ds.phoneme_vocab, fh, ensure_ascii=False, indent=2)
    print(f"Saved phoneme vocab ({len(train_ds.phoneme_vocab)}) to {args.output_dir}/phoneme_vocab.json")

    train_loader = DataLoader(train_ds, batch_size=args.per_device_batch_size, shuffle=True, collate_fn=lambda b: collate_batch(b, tokenizer, max_length=args.max_length))
    val_loader = DataLoader(val_ds, batch_size=args.per_device_batch_size, shuffle=False, collate_fn=lambda b: collate_batch(b, tokenizer, max_length=args.max_length)) if val_ds else None

    # prepare optimizer: only update trainable params
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps)

    print("***** Running training *****")
    print(f"Num Epochs = {args.epochs}")
    print(f"Train samples = {len(train_ds)}; Val samples = {len(val_ds) if val_ds else 0}")
    print(f"Total training steps = {total_steps}")

    global_step = 0
    best_loss = float('inf')

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar, start=1):
            # batch is a tokenized batch dict
            inputs = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=args.fp16 and torch.cuda.is_available()):
                outputs = model(**inputs, labels=inputs["input_ids"])  # causal LM label setup
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"ckpt-step{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    # save model & tokenizer
                    try:
                        model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                    except Exception as e:
                        print("Warning: saving pretrained failed:", e)
                    print(f"Saved checkpoint to {ckpt_dir}")

                pbar.set_postfix({"loss": f"{(epoch_loss / step):.4f}", "step": global_step})

        # end epoch
        # simple validation: compute average loss on val set (if present)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for vb in tqdm(val_loader, desc="Validation"):
                    vb = {k: v.to(device) for k, v in vb.items()}
                    with torch.cuda.amp.autocast(enabled=args.fp16 and torch.cuda.is_available()):
                        out = model(**vb, labels=vb["input_ids"])  # labels for LM loss
                        val_loss += out.loss.item()
                        n += 1
            val_loss = val_loss / max(1, n)
            print(f"Epoch {epoch} validation loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_dir = os.path.join(args.output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                try:
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                except Exception as e:
                    print("Warning: saving best model failed:", e)
                print(f"Saved best model to {best_dir}")

    print("Training finished.")

# ----------------------------------------------------------------------------
# HARDCODED ARGUMENTS SETUP
# ----------------------------------------------------------------------------

def get_hardcoded_args():
    """Returns a namespace object with all training arguments hardcoded."""
    
    # === 请在这里修改您的设置 ===
    HARDCODED_SETTINGS = {
        # 必需参数
        "train_tsv": "train.tsv",
        "val_tsv": "val.tsv", # 设置为 None 如果没有验证集
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct", # 请确认您有权限访问
        "output_dir": "outputs",
        
        # 训练参数
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "epochs": 3,
        "lr": 2e-4,
        "weight_decay": 0.0,
        "save_steps": 500,
        "max_length": 512,
        
        # 优化/内存参数 (建议保持启用以解决 OOM)
        "fp16": False, # 如果显存紧张且没有使用 4bit，可以尝试设置为 True
        "use_4bit": True, # 启用 4-bit 量化 (推荐)
        "use_lora": True, # 启用 LoRA (推荐)
        "gradient_checkpointing": True, # 启用梯度检查点 (推荐)
        
        # LoRA 配置
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": ["q_proj", "v_proj"], # LLaMA 常用目标
        
        # 额外的 phonemes
        "extra_phonemes": "", # 示例: "AH0,AA1,AE2" (逗号分隔)
    }
    # ==============================

    # Create a SimpleNamespace object to mimic argparse behavior
    args = SimpleNamespace(**HARDCODED_SETTINGS)
    
    # Normalize extra_phonemes (if the default comma was used)
    if args.extra_phonemes == ",":
        args.extra_phonemes = ""
        
    return args


if __name__ == "__main__":
    args = get_hardcoded_args()
    train_loop(args)
