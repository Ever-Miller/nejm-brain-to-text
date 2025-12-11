#!/usr/bin/env python3
# coding: utf-8

"""
Phoneme -> Sentence training script
Usage example:
    python train_phoneme2sentence.py \
      --data_path data/train.tsv \
      --val_path data/val.tsv \
      --model_name gpt2 \
      --output_dir checkpoints/phoneme2sent \
      --epochs 3 \
      --batch_size 8
"""

import os
import math
import argparse
from typing import List, Dict
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType

from tqdm import tqdm

# ---------------------
# Dataset
# ---------------------
class PhonemeTextDataset(Dataset):
    """
    Expects tsv with two columns: phonemes \t sentence
    phonemes: space-separated tokens, e.g. "S IY K R IH T"
    """
    def __init__(self, path: str, phoneme_vocab: Dict[str,int]=None, max_phoneme_len=512):
        self.samples = []
        self.max_phoneme_len = max_phoneme_len
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                # support tsv or jsonl
                if '\t' in line:
                    p,s = line.split('\t', 1)
                else:
                    # fallback: assume two columns split by |||
                    parts = line.split('|||')
                    if len(parts)>=2:
                        p,s = parts[0].strip(), parts[1].strip()
                    else:
                        continue
                phonemes = p.strip().split()
                self.samples.append((phonemes, s.strip()))
        # build phoneme vocab if not provided
        if phoneme_vocab is None:
            self.phoneme_vocab = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}
            idx = max(self.phoneme_vocab.values())+1
            for phon_seq, _ in self.samples:
                for ph in phon_seq:
                    if ph not in self.phoneme_vocab:
                        self.phoneme_vocab[ph] = idx
                        idx += 1
        else:
            self.phoneme_vocab = phoneme_vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        phonemes, sentence = self.samples[idx]
        phoneme_ids = [self.phoneme_vocab.get(p, self.phoneme_vocab.get("<unk>")) for p in phonemes]
        # truncate
        if len(phoneme_ids) > self.max_phoneme_len - 2:
            phoneme_ids = phoneme_ids[:self.max_phoneme_len-2]
        # add bos/eos optionally
        phoneme_ids = [self.phoneme_vocab["<bos>"]] + phoneme_ids + [self.phoneme_vocab["<eos>"]]
        return {"phonemes": phoneme_ids, "sentence": sentence}

def collate_fn(batch, tokenizer, phoneme_pad_id=0, max_target_len=256):
    phoneme_seqs = [torch.tensor(x["phonemes"], dtype=torch.long) for x in batch]
    phoneme_lens = [len(x) for x in phoneme_seqs]
    max_ph_len = max(phoneme_lens)
    phoneme_padded = torch.full((len(batch), max_ph_len), phoneme_pad_id, dtype=torch.long)
    for i,seq in enumerate(phoneme_seqs):
        phoneme_padded[i,:seq.size(0)] = seq

    sentences = [x["sentence"] for x in batch]
    # Use tokenizer to encode target sentences (we will convert to embeddings later)
    tokenized = tokenizer(sentences, padding=True, truncation=True, max_length=max_target_len, return_tensors="pt")
    return {
        "phoneme_ids": phoneme_padded,
        "phoneme_lens": torch.tensor(phoneme_lens, dtype=torch.long),
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"]
    }

# ---------------------
# Encoder module
# ---------------------
class PhonemeEncoder(nn.Module):
    def __init__(self, vocab_size:int, embed_dim=512, num_layers=8, num_heads=8, ffn_dim=2048, max_len=1024, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed_dim = embed_dim
        self.max_len = max_len

    def forward(self, phoneme_ids, attention_mask=None):
        # phoneme_ids: [B, L]
        b,l = phoneme_ids.shape
        h = self.embed(phoneme_ids) + self.pos_emb[:, :l, :].to(phoneme_ids.device)
        # if attention_mask provided, compute key_padding_mask for transformer
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for pad
            key_padding_mask = (~(attention_mask.bool())).to(phoneme_ids.device)  # True at pad positions
        else:
            key_padding_mask = None
        out = self.encoder(h, src_key_padding_mask=key_padding_mask)  # [B,L,embed_dim]
        return out

# ---------------------
# Full wrapper model (handles bridge & loss)
# ---------------------
class Phoneme2SentenceModel(nn.Module):
    def __init__(self, encoder:PhonemeEncoder, decoder_model:AutoModelForCausalLM, decoder_hidden_size:int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder_model
        # projection from encoder embed -> decoder hidden size
        self.proj = nn.Linear(encoder.embed_dim, decoder_hidden_size)
        self.decoder_hidden_size = decoder_hidden_size

    def forward(self, phoneme_ids, phoneme_attention_mask, target_input_ids, target_attention_mask):
        """
        phoneme_ids: [B, Lp]
        phoneme_attention_mask: [B, Lp] (1 for real tokens)
        target_input_ids: [B, Lt] (token ids of target sentences)
        target_attention_mask: [B, Lt]
        """
        device = phoneme_ids.device
        encoder_out = self.encoder(phoneme_ids, phoneme_attention_mask)  # [B, Lp, enc_dim]
        prefix_embeds = self.proj(encoder_out)  # [B, Lp, dec_hidden]

        # obtain target token embeddings from decoder's input embeddings
        input_embed_layer = self.decoder.get_input_embeddings()  # Embedding layer
        target_embeds = input_embed_layer(target_input_ids)  # [B, Lt, dec_hidden]

        # concat prefix and target embeddings to form full inputs_embeds
        inputs_embeds = torch.cat([prefix_embeds, target_embeds], dim=1)  # [B, Lp+Lt, dec_hidden]

        # build labels: prefix positions -> -100 (ignored), target positions -> token ids
        batch_size = target_input_ids.size(0)
        prefix_len = prefix_embeds.size(1)
        labels = torch.full((batch_size, prefix_len), -100, dtype=torch.long, device=device)
        labels = torch.cat([labels, target_input_ids], dim=1)  # [B, Lp+Lt]

        # build attention mask for full input
        full_attention = torch.cat([phoneme_attention_mask, target_attention_mask], dim=1)  # [B, Lp+Lt]

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            labels=labels,
        )
        # outputs.loss is the causal LM loss
        return outputs

# ---------------------
# Utility: freeze decoder except LoRA (peft will re-enable trainable params)
# ---------------------
def freeze_model_params(model):
    for p in model.parameters():
        p.requires_grad = False

# ---------------------
# Training loop
# ---------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ---------------------
    # Prepare tokenizer & decoder
    # ---------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # load causal lm
    decoder = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16 if device.type=='cuda' else torch.float32, low_cpu_mem_usage=True)
    # resize token embeddings if we added pad token
    decoder.resize_token_embeddings(len(tokenizer))

    # ---------------------
    # Prepare datasets
    # ---------------------
    train_ds = PhonemeTextDataset(args.data_path, max_phoneme_len=args.max_phoneme_len)
    val_ds = PhonemeTextDataset(args.val_path, phoneme_vocab=train_ds.phoneme_vocab, max_phoneme_len=args.max_phoneme_len) if args.val_path else None
    print("Phoneme vocab size:", len(train_ds.phoneme_vocab))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, phoneme_pad_id=train_ds.phoneme_vocab["<pad>"], max_target_len=args.max_target_len))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, phoneme_pad_id=train_ds.phoneme_vocab["<pad>"], max_target_len=args.max_target_len)) if val_ds else None

    # ---------------------
    # Build encoder + wrapper
    # ---------------------
    encoder = PhonemeEncoder(vocab_size=len(train_ds.phoneme_vocab), embed_dim=args.encoder_embed_dim, num_layers=args.encoder_layers, num_heads=args.encoder_heads, ffn_dim=args.encoder_ffn_dim, max_len=args.max_phoneme_len)
    # move decoder to device before hooking into wrapper
    decoder.to(device)
    # freeze decoder parameters
    freeze_model_params(decoder)

    model = Phoneme2SentenceModel(encoder, decoder, decoder_hidden_size=decoder.config.hidden_size)
    model.to(device)

    # ---------------------
    # Setup LoRA (PEFT) on decoder
    # ---------------------
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules  # may need to adjust for your decoder
    )
    # apply LoRA to the decoder model inside wrapper
    # Note: get_peft_model expects the model object to be the HF model; we'll apply to model.decoder
    model.decoder = get_peft_model(model.decoder, peft_config)
    # now move peft parameters to device (if not)
    model.decoder.to(device)

    # ---------------------
    # Optimizer: only LoRA (and encoder + projection) trainable?
    # Two options:
    # 1) Train encoder + LoRA => update both
    # 2) Freeze encoder, only LoRA => faster, less overfit
    # We'll train encoder + LoRA
    # ---------------------
    # ensure encoder params trainable
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.proj.parameters():
        p.requires_grad = True

    # collect trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable_params))

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03*total_steps), num_training_steps=total_steps)

    # ---------------------
    # Training loop
    # ---------------------
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for batch in pbar:
            phoneme_ids = batch["phoneme_ids"].to(device)
            phoneme_mask = (phoneme_ids != train_ds.phoneme_vocab["<pad>"]).long().to(device)
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["attention_mask"].to(device)

            outputs = model(phoneme_ids=phoneme_ids, phoneme_attention_mask=phoneme_mask, target_input_ids=input_ids, target_attention_mask=input_mask)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{running_loss/global_step:.4f}"})

            if global_step % args.save_steps == 0:
                # save checkpoint
                save_dir = os.path.join(args.output_dir, f"ckpt-step{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                # save encoder + peft adapter
                torch.save(model.encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
                torch.save(model.proj.state_dict(), os.path.join(save_dir, "proj.pt"))
                # save peft adapter
                model.decoder.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print("Saved checkpoint to", save_dir)

        # --- validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    phoneme_ids = batch["phoneme_ids"].to(device)
                    phoneme_mask = (phoneme_ids != train_ds.phoneme_vocab["<pad>"]).long().to(device)
                    input_ids = batch["input_ids"].to(device)
                    input_mask = batch["attention_mask"].to(device)

                    outputs = model(phoneme_ids=phoneme_ids, phoneme_attention_mask=phoneme_mask, target_input_ids=input_ids, target_attention_mask=input_mask)
                    l = outputs.loss.item()
                    val_loss += l
                    n += 1
            val_loss = val_loss / max(1, n)
            print(f"Epoch {epoch} validation loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_dir = os.path.join(args.output_dir, "best")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.encoder.state_dict(), os.path.join(save_dir, "encoder.pt"))
                torch.save(model.proj.state_dict(), os.path.join(save_dir, "proj.pt"))
                model.decoder.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print("Saved best model to", save_dir)

    print("Training done.")

# ---------------------
# Inference / generation helper
# ---------------------
@torch.no_grad()
def generate_from_phonemes(checkpoint_dir, phoneme_seq: List[str], tokenizer, device, max_new_tokens=64):
    # load saved components
    # load phoneme vocab
    import json
    vocab_path = os.path.join(checkpoint_dir, "phoneme_vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError("phoneme_vocab.json not found in checkpoint_dir. You must save the phoneme vocab yourself.")
    phoneme_vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))

    # create encoder
    encoder_state = torch.load(os.path.join(checkpoint_dir, "encoder.pt"), map_location=device)
    # recreate encoder with correct shapes (we assume defaults)
    encoder = PhonemeEncoder(vocab_size=len(phoneme_vocab))
    encoder.load_state_dict(encoder_state)
    encoder.to(device)
    encoder.eval()

    # load decoder + peft
    decoder = AutoModelForCausalLM.from_pretrained(checkpoint_dir, torch_dtype=torch.float16 if device.type=='cuda' else torch.float32)
    decoder.to(device)
    decoder.eval()

    # load proj
    proj_state = torch.load(os.path.join(checkpoint_dir, "proj.pt"), map_location=device)
    proj = nn.Linear(encoder.embed_dim, decoder.config.hidden_size).to(device)
    proj.load_state_dict(proj_state)
    proj.eval()

    # encode phonemes
    phoneme_ids = [phoneme_vocab.get("<bos>",1)] + [phoneme_vocab.get(p,phoneme_vocab.get("<unk>")) for p in phoneme_seq] + [phoneme_vocab.get("<eos>",2)]
    ph_tensor = torch.tensor([phoneme_ids], dtype=torch.long).to(device)
    ph_mask = (ph_tensor != phoneme_vocab.get("<pad>",0)).long().to(device)
    enc_out = encoder(ph_tensor, ph_mask)  # [1, Lp, enc_dim]
    prefix_embeds = proj(enc_out)  # [1, Lp, dec_hidden]

    # prepare generation loop: we will feed prefix_embeds + generated tokens' embeddings iteratively
    generated = []
    # initial inputs_embeds = prefix_embeds
    cur_inputs = prefix_embeds  # [1, Lp, dec_hidden]
    cur_attention = torch.ones(1, cur_inputs.size(1), dtype=torch.long, device=device)
    for step in range(max_new_tokens):
        # pass through decoder and sample logits for next token
        outputs = decoder(inputs_embeds=cur_inputs, attention_mask=cur_attention)
        # outputs.logits: [1, seq_len, vocab_size]
        next_logits = outputs.logits[:, -1, :]  # [1, vocab]
        next_token = torch.argmax(next_logits, dim=-1)  # greedy
        if next_token.item() == tokenizer.eos_token_id:
            break
        generated.append(next_token.item())
        # append embedding of next_token to inputs_embeds for next step
        next_emb = decoder.get_input_embeddings()(next_token.unsqueeze(0))  # [1, 1, hidden]
        cur_inputs = torch.cat([cur_inputs, next_emb], dim=1)
        cur_attention = torch.cat([cur_attention, torch.ones(1,1,device=device)], dim=1)

    gen_text = tokenizer.decode(generated, skip_special_tokens=True)
    return gen_text

# ---------------------
# CLI
# ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="train tsv path")
    p.add_argument("--val_path", default=None, help="validation tsv path")
    p.add_argument("--model_name", default="gpt2", help="HF causal LM model name")
    p.add_argument("--output_dir", default="outputs", help="where to save checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--max_phoneme_len", type=int, default=256)
    p.add_argument("--max_target_len", type=int, default=128)
    p.add_argument("--encoder_embed_dim", type=int, default=512)
    p.add_argument("--encoder_layers", type=int, default=8)
    p.add_argument("--encoder_heads", type=int, default=8)
    p.add_argument("--encoder_ffn_dim", type=int, default=2048)
    # LoRA params
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    # default target modules (may vary by model; adjust if training errors)
    p.add_argument("--lora_target_modules", nargs="+", default=["q_proj","v_proj"], help="target modules for LoRA (model specific)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
