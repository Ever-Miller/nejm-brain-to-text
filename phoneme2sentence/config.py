# config.py
import os

# ==========================
# 1. Phonemes and Vocab (Single Source of Truth)
# ==========================
# Base phoneme list
PHONEMES_LIST = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
]

# Special Tokens
TOKEN_PAD = "<pad>"
TOKEN_SOS = "<sos>"
TOKEN_EOS = "<eos>"
TOKEN_MASK = "[MASK]"
TOKEN_UNK = "<unk>" # Added UNK for safety

SPECIAL_TOKENS = [TOKEN_PAD, TOKEN_SOS, TOKEN_EOS, TOKEN_MASK, TOKEN_UNK]
PHONEME_VOCAB = SPECIAL_TOKENS + PHONEMES_LIST

# Mapping dictionaries
phoneme_to_id = {p: i for i, p in enumerate(PHONEME_VOCAB)}
id_to_phoneme = {i: p for p, i in phoneme_to_id.items()}

VOCAB_SIZE_PHONEME = len(PHONEME_VOCAB)
PAD_ID_PHONEME = phoneme_to_id[TOKEN_PAD]
SOS_ID = phoneme_to_id[TOKEN_SOS]
EOS_ID = phoneme_to_id[TOKEN_EOS]
MASK_ID_PHONEME = phoneme_to_id[TOKEN_MASK]
UNK_ID = phoneme_to_id[TOKEN_UNK]

# ==========================
# 2. Model Hyperparameters
# ==========================
PLM_MODEL_NAME = "gpt2" # Decoder (English)
MAX_P_LEN = 128         # Encoder Input Length
MAX_S_LEN = 64          # Decoder Output Length

# BERT Encoder Config (Must match train_mpm)
ENCODER_HIDDEN_SIZE = 768
ENCODER_NUM_LAYERS = 6
ENCODER_ATTN_HEADS = 12  # 768 / 12 = 64 per head

# ==========================
# 3. Training & Paths
# ==========================
DEBUG_MODE = False  # Set True for quick testing

DATA_DIR = "./data"
MPM_OUTPUT_DIR = "./checkpoints/mpm_pretrain"
BEST_ENCODER_PATH = "./checkpoints/best_mpm_encoder"
P2S_OUTPUT_DIR = "./checkpoints/p2s_finetune"
FINAL_MODEL_DIR = "./output_model"

# Hyperparams
BATCH_SIZE_MPM = 128
BATCH_SIZE_P2S = 32
EPOCHS_MPM = 30
EPOCHS_P2S = 35
LR_MPM = 1e-4
LR_P2S = 5e-5

# LoRA Config
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

SEED = 42