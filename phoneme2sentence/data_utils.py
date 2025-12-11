# data_utils.py
import os
import torch
import json
import pandas as pd
from typing import List, Dict, Union
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
import config

# ---------- 1. Initialize Dec Tokenizer (GPT2) ----------
gpt_tokenizer = AutoTokenizer.from_pretrained(config.PLM_MODEL_NAME)
# GPT2 has no default pad token, usually set to eos
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token 

# ---------- 2. Phoneme Tokenizer (Custom) ----------
class PhonemeTokenizer:
    def __init__(self):
        self.vocab = config.PHONEME_VOCAB
        self.token_to_id = config.phoneme_to_id
        self.id_to_token = config.id_to_phoneme
        
        self.pad_token_id = config.PAD_ID_PHONEME
        self.mask_token_id = config.MASK_ID_PHONEME
        self.sos_token_id = config.SOS_ID
        self.eos_token_id = config.EOS_ID
        self.unk_token_id = config.UNK_ID
        self.vocab_size = config.VOCAB_SIZE_PHONEME

    def encode(self, phoneme_str: str, max_length: int = config.MAX_P_LEN, add_special_tokens=True):
        """
        Input: "HH AH L OW" or list of phonemes
        """
        if isinstance(phoneme_str, str):
            parts = phoneme_str.strip().split()
        else:
            parts = phoneme_str
            
        # Mapping
        ids = [self.token_to_id.get(p, self.unk_token_id) for p in parts]
        
        if add_special_tokens:
            ids = [self.sos_token_id] + ids + [self.eos_token_id]
            
        # Truncate
        if len(ids) > max_length:
            ids = ids[:max_length]
            if add_special_tokens:
                ids[-1] = self.eos_token_id # Ensure EOS is at end

        # Pad (Manual padding)
        attention_mask = [1] * len(ids)
        pad_len = max_length - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            
        return {"input_ids": ids, "attention_mask": attention_mask}

    def decode(self, ids: List[int], skip_special_tokens=True):
        tokens = []
        for i in ids:
            if skip_special_tokens and i in [self.pad_token_id, self.sos_token_id, self.eos_token_id]:
                continue
            tokens.append(self.id_to_token.get(i, config.TOKEN_UNK))
        return " ".join(tokens)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "vocab.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, indent=2)

# Global Instance
phoneme_tokenizer = PhonemeTokenizer()

# ---------- 3. Preprocessing Functions ----------

def preprocess_mpm(examples):
    # MPM needs only phoneme encoding
    batch_input_ids = []
    batch_attention_mask = []
    
    for p_str in examples["phonemes"]:
        encoded = phoneme_tokenizer.encode(p_str, max_length=config.MAX_P_LEN)
        batch_input_ids.append(encoded["input_ids"])
        batch_attention_mask.append(encoded["attention_mask"])
        
    return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask}

def preprocess_p2s(examples):
    # 1. Encoder Inputs (Phonemes)
    model_inputs = preprocess_mpm(examples)
    
    # 2. Decoder Inputs (Sentences)
    labels = gpt_tokenizer(
        text_target=examples["sentence"], 
        max_length=config.MAX_S_LEN, 
        truncation=True,
        padding="max_length"
    )

    # Convert pad tokens in labels to -100 (ignored in loss)
    label_ids = labels["input_ids"]
    cleaned_labels = []
    for seq in label_ids:
        cleaned_seq = [l if l != gpt_tokenizer.pad_token_id else -100 for l in seq]
        cleaned_labels.append(cleaned_seq)
        
    model_inputs["labels"] = cleaned_labels
    return model_inputs

# ---------- 4. Collators ----------

class PhonemeMaskingCollator:
    """For MPM Training"""
    def __init__(self, mask_prob=0.15):
        self.mask_prob = mask_prob
        self.mask_token_id = config.MASK_ID_PHONEME
        self.pad_token_id = config.PAD_ID_PHONEME
        self.vocab_size = config.VOCAB_SIZE_PHONEME

    def __call__(self, features):
        # Convert to tensor
        input_ids = torch.tensor([f["input_ids"] for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f["attention_mask"] for f in features], dtype=torch.long)
        
        labels = input_ids.clone()
        
        # Probability Matrix
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        special_tokens_mask = (input_ids == self.pad_token_id) | \
                              (input_ids == config.SOS_ID) | \
                              (input_ids == config.EOS_ID)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100 # Ignore non-masked loss

        # 80% [MASK], 10% Random, 10% Original
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.mask_token_id

        # Random replacement
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        if indices_random.any():
            random_words = torch.randint(0, self.vocab_size, labels.shape, dtype=torch.long)
            input_ids[indices_random] = random_words[indices_random]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ---------- 5. Data Loading Helper ----------
def load_and_process_data(mode="mpm"):
    """
    Loads train.tsv/val.tsv. Expects columns: 'phonemes', 'sentence'
    """
    train_path = os.path.join(config.DATA_DIR, "train.tsv")
    val_path = os.path.join(config.DATA_DIR, "val.tsv")

    # Fallback to creating dummy data if not exists
    if not os.path.exists(train_path): 
        print(f"Warning: {train_path} not found.")
        return None, None

    df_train = pd.read_csv(train_path, sep="\t").dropna().astype(str)
    df_val = pd.read_csv(val_path, sep="\t").dropna().astype(str)

    if config.DEBUG_MODE:
        print("!!! DEBUG MODE: Using 20 samples only !!!")
        df_train = df_train.head(20)
        df_val = df_val.head(20)

    ds_train = Dataset.from_pandas(df_train)
    ds_val = Dataset.from_pandas(df_val)

    process_fn = preprocess_mpm if mode == "mpm" else preprocess_p2s
    cols_to_remove = ["phonemes", "sentence"] if "sentence" in ds_train.column_names else ["phonemes"]
    
    # Keep only necessary columns for processing
    cols_in_ds = ds_train.column_names
    cols_to_remove = [c for c in cols_to_remove if c in cols_in_ds]

    print(f"Processing dataset for mode: {mode}...")
    ds_train = ds_train.map(process_fn, batched=True, remove_columns=cols_to_remove)
    ds_val = ds_val.map(process_fn, batched=True, remove_columns=cols_to_remove)
    
    return ds_train, ds_val