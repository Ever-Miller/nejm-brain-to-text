import pandas as pd
import torch
import numpy as np
from datasets import Dataset
import evaluate
import os 
import glob

# LoRA/PEFT 库导入
from peft import LoraConfig, get_peft_model, TaskType
from transformers import PreTrainedTokenizer 
from transformers import (
    AutoTokenizer, 
    BertConfig, 
    BertModel,
    BertForMaskedLM, 
    GPT2Config, 
    GPT2LMHeadModel, 
    EncoderDecoderConfig, 
    EncoderDecoderModel,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling, 
    EarlyStoppingCallback
)

# ==========================================
# 1. 全局配置 & 音素定义
# ==========================================

PHONEMES_LIST = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "BLANK", 
    "[MASK]" 
]

SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '.', '|'] 
PHONEME_VOCAB = SPECIAL_TOKENS + PHONEMES_LIST
phoneme_to_id = {p: i for i, p in enumerate(PHONEME_VOCAB)}
id_to_phoneme = {i: p for p, i in phoneme_to_id.items()}

VOCAB_SIZE_PHONEME = len(PHONEME_VOCAB)
PAD_ID_PHONEME = phoneme_to_id['<pad>']
SOS_ID = phoneme_to_id['<sos>']
EOS_ID = phoneme_to_id['<eos>']

try:
    MASK_ID_PHONEME = phoneme_to_id['[MASK]']
except KeyError:
    MASK_ID_PHONEME = PAD_ID_PHONEME 

# 模型与训练超参数
PLM_MODEL_NAME = "gpt2"      
MAX_P_LEN = 128              
MAX_S_LEN = 50               
ENCODER_HIDDEN_SIZE = 256    # 已优化：减小 Hidden Size
NUM_EPOCHS = 50              

# MPM 预训练配置
MPM_PRETRAIN_EPOCHS = 10     
MPM_OUTPUT_DIR = "./mpm_pretrain_checkpoints"

# ==========================================
# 2. 数据处理函数
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# --- DummyTokenizer (修复版) ---
class DummyPhonemeTokenizer:
    def __init__(self, pad_id, mask_id, sos_id, eos_id, vocab_size):
        self.pad_token_id = pad_id
        self.mask_token_id = mask_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.mask_token = "[MASK]"
        self.pad_token = "<pad>"
        self.all_special_ids = [pad_id, mask_id, sos_id, eos_id]

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return {"[MASK]": self.mask_token_id, "<pad>": self.pad_token_id}

    def save_pretrained(self, save_directory, **kwargs):
        pass

    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None):
        import torch
        input_ids = [example['input_ids'] for example in encoded_inputs]
        if return_tensors == 'pt':
            batch = {'input_ids': torch.tensor(input_ids, dtype=torch.long)}
            if len(encoded_inputs) > 0 and 'attention_mask' in encoded_inputs[0]:
                attention_masks = [example['attention_mask'] for example in encoded_inputs]
                batch['attention_mask'] = torch.tensor(attention_masks, dtype=torch.long)
            return batch
        return encoded_inputs

    def get_special_tokens_mask(self, token_ids_0, already_has_special_tokens=False):
        return [1 if token in self.all_special_ids else 0 for token in token_ids_0]

    def convert_tokens_to_ids(self, token):
        if token == self.mask_token: return self.mask_token_id
        if token == self.pad_token: return self.pad_token_id
        return 0 

# --- Collator (修复版) ---
class PhonemeMaskingDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, *args, **kwargs):
        dummy_tokenizer = DummyPhonemeTokenizer(
            pad_id=PAD_ID_PHONEME, 
            mask_id=MASK_ID_PHONEME, 
            sos_id=SOS_ID, 
            eos_id=EOS_ID, 
            vocab_size=VOCAB_SIZE_PHONEME
        )
        super().__init__(dummy_tokenizer, mlm=True, mlm_probability=0.15, **kwargs)
        self.pad_token_id = PAD_ID_PHONEME

# --- 预处理函数 ---
def preprocess_function(example):
    phoneme_seq = str(example['phonemes']).split()
    cleaned_phoneme_seq = [p for p in phoneme_seq if p not in ['BLANK', '[MASK]']]
    
    p_ids = [SOS_ID] + \
            [phoneme_to_id.get(p, PAD_ID_PHONEME) for p in cleaned_phoneme_seq] + \
            [EOS_ID]
    
    if len(p_ids) > MAX_P_LEN:
        p_ids = p_ids[:MAX_P_LEN]
        p_ids[-1] = EOS_ID 
        
    attention_mask_p = [1] * len(p_ids)
    
    padding_len = MAX_P_LEN - len(p_ids)
    if padding_len > 0:
        p_ids.extend([PAD_ID_PHONEME] * padding_len)
        attention_mask_p.extend([0] * padding_len)

    s_tokenized = tokenizer(
        example['sentence'], 
        max_length=MAX_S_LEN, 
        padding="max_length", 
        truncation=True
    )
    
    labels = s_tokenized['input_ids'].copy()
    labels = [l if l != tokenizer.pad_token_id else -100 for l in labels]

    return {
        'input_ids': p_ids,          
        'attention_mask': attention_mask_p, 
        'labels': labels,            
    }

def mpm_preprocess_function(example):
    # 已优化：添加 Attention Mask
    phoneme_seq = str(example['phonemes']).split()
    cleaned_phoneme_seq = [p for p in phoneme_seq if p not in ['BLANK', '[MASK]']]
    
    p_ids = [SOS_ID] + \
            [phoneme_to_id.get(p, PAD_ID_PHONEME) for p in cleaned_phoneme_seq] + \
            [EOS_ID]
    
    if len(p_ids) > MAX_P_LEN:
        p_ids = p_ids[:MAX_P_LEN]
        p_ids[-1] = EOS_ID 
    
    attention_mask = [1] * len(p_ids)
    
    padding_len = MAX_P_LEN - len(p_ids)
    if padding_len > 0:
        p_ids.extend([PAD_ID_PHONEME] * padding_len)
        attention_mask.extend([0] * padding_len)

    return {
        'input_ids': p_ids,
        'attention_mask': attention_mask
    }

# ==========================================
# 3. 数据加载
# ==========================================

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}. 请确保 train.tsv 和 val.tsv 存在。")
    df = pd.read_csv(file_path, sep='\t').dropna().reset_index(drop=True)
    return Dataset.from_pandas(df)

try:
    train_dataset_raw = load_data('train.tsv')
    val_dataset_raw = load_data('val.tsv')
except FileNotFoundError as e:
    print(f"致命错误: {e}")
    exit()

tokenized_train_p2s = train_dataset_raw.map(preprocess_function, remove_columns=['phonemes', 'sentence'])
tokenized_val_p2s = val_dataset_raw.map(preprocess_function, remove_columns=['phonemes', 'sentence'])
tokenized_train_mpm = train_dataset_raw.map(mpm_preprocess_function, remove_columns=['phonemes', 'sentence'])
tokenized_val_mpm = val_dataset_raw.map(mpm_preprocess_function, remove_columns=['phonemes', 'sentence'])

# ==========================================
# 4. 阶段一：MPM 预训练函数
# ==========================================
def run_mpm_pretraining(train_dataset, val_dataset):
    print("="*50)
    print("🚀 阶段一：开始 MPM (Masked Phoneme Modeling) 预训练 (优化版)")
    print("="*50)

    # 4.1 Encoder 配置 (加深网络)
    encoder_config = BertConfig(
        vocab_size=VOCAB_SIZE_PHONEME, 
        hidden_size=256,         
        num_hidden_layers=8,          # 6 -> 8 层，增加推理深度
        num_attention_heads=4,
        intermediate_size=256 * 4,
        pad_token_id=PAD_ID_PHONEME
    )
    
    mpm_model = BertForMaskedLM(config=encoder_config)
    mpm_model.bert.embeddings.word_embeddings = torch.nn.Embedding(
        VOCAB_SIZE_PHONEME, 256, padding_idx=PAD_ID_PHONEME
    )

    # 4.2 MPM 训练参数 (增加训练量)
    mpm_args = TrainingArguments(
        output_dir=MPM_OUTPUT_DIR,
        num_train_epochs=50,                # 10 -> 50: 预训练需要更多轮次才能收敛
        per_device_train_batch_size=16,     # 尝试稍微增大单卡 Batch (如果显存不够改回 8)
        gradient_accumulation_steps=1,      # 移除累积，让参数更新更频繁
        learning_rate=1e-4,                 # 稍微降低一点 LR，配合更多的 Epoch
        warmup_ratio=0.1,                   # 使用比例 Warmup
        weight_decay=0.01,
        logging_steps=50,
        
        eval_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2, 
        
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    mpm_data_collator = PhonemeMaskingDataCollator() 

    mpm_trainer = Trainer(
        model=mpm_model,
        args=mpm_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=mpm_data_collator,
    )

    # 4.5 开始训练
    mpm_trainer.train()
    
    print("MPM 预训练完成。加载最佳 Encoder 权重...")
    
    # --- 崩溃修复逻辑 ---
    # Trainer 如果 load_best_model_at_end=True，训练结束时内存里的 model 已经是最佳模型了。
    # 我们不需要手动去 load state dict，除非你想双重保险。
    # 但为了解决之前的报错，我们加上 safetensors 的判断逻辑。
    
    best_ckpt_path = mpm_trainer.state.best_model_checkpoint
    if best_ckpt_path:
        print(f"最佳检查点路径: {best_ckpt_path}")
        # 尝试直接使用 Trainer 当前的模型 (它已经加载了最佳权重)
        # 这是一个小技巧，通常 Trainer 训练完会自动回滚到最佳权重
    else:
        print("未找到最佳检查点，使用最终模型权重。")

    # 只要不报错，直接返回 mpm_model.bert 即可
    # 因为 load_best_model_at_end=True 保证了 mpm_model 现在就是最佳状态
    return mpm_model.bert

# ==========================================
# 5. 阶段二：P2S 微调设置
# ==========================================

def compute_metrics(eval_pred):
    # 占位符
    predictions, labels = eval_pred
    return {"loss_placeholder": 0.0} 

BATCH_SIZE = 8
STEPS_PER_EPOCH = len(tokenized_train_p2s) // BATCH_SIZE
EVAL_INTERVAL_EPOCHS = 2 
EVAL_STEPS = STEPS_PER_EPOCH * EVAL_INTERVAL_EPOCHS

training_args_p2s = Seq2SeqTrainingArguments(
    output_dir="./p2s_checkpoints_lora_trainable_encoder", 
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    save_strategy="steps",
    save_steps=EVAL_STEPS,
    save_total_limit=10, 
    predict_with_generate=True,
    generation_max_length=MAX_S_LEN,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

data_collator_p2s = DataCollatorForSeq2Seq(
    tokenizer, 
    model=None, 
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

# ==========================================
# 6. 开始执行训练流程
# ==========================================

if __name__ == "__main__":
    
    # --- 阶段一：MPM 预训练 ---
    pretrain_encoder = run_mpm_pretraining(tokenized_train_mpm, tokenized_val_mpm)
    
    # --- 阶段二：P2S 微调 ---
    print("\n" + "="*50)
    print("📝 阶段二：开始 P2S (Phoneme-to-Text) 微调 (LoRA + Encoder 可训练)")
    print("="*50)
    
    encoder_config = pretrain_encoder.config 
    encoder = pretrain_encoder 
    # ⚠️ 关键点：我们保持 encoder 的 requires_grad=True，使其可训练。

    decoder_config = GPT2Config.from_pretrained(PLM_MODEL_NAME)
    decoder_config.add_cross_attention = True
    decoder_config.is_decoder = True 
    decoder = GPT2LMHeadModel.from_pretrained(PLM_MODEL_NAME, config=decoder_config)
    
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    model = EncoderDecoderModel(config=config, encoder=encoder, decoder=decoder)

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.config.max_length = MAX_S_LEN
    model.config.min_length = 2
    model.config.no_repeat_ngram_size = 3
    model.config.early_stopping = True
    model.config.length_penalty = 1.0
    model.config.num_beams = 4 

    # ==========================================
    # 🔥 LoRA 核心修改：针对 GPT-2 结构 🔥
    # ==========================================
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        # ⚠️ 修改：GPT-2 使用 c_attn 卷积层，而不是 q_proj/v_proj
        target_modules=["c_attn"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_TO_SEQ_LM,
    )
    
    peft_decoder = get_peft_model(model.decoder, lora_config)
    model.decoder = peft_decoder
    
    print("\n--- 模型参数信息 ---")
    peft_decoder.print_trainable_parameters()
    print("Encoder (BERT) 参数状态：")
    
    trainable_params_encoder = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    trainable_params_decoder = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())

    print(f"  Encoder (可训练): {trainable_params_encoder:,} (保持可训练)")
    print(f"  Decoder (LoRA 可训练): {trainable_params_decoder:,}")
    print(f"总可训练参数: {trainable_params_encoder + trainable_params_decoder:,}")
    print(f"总参数: {all_params:,}")
    print("------------------------\n")
    
    trainer_p2s = Seq2SeqTrainer(
        model=model,
        args=training_args_p2s,
        train_dataset=tokenized_train_p2s,
        eval_dataset=tokenized_val_p2s,
        tokenizer=tokenizer,
        data_collator=data_collator_p2s,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    print(f"Starting P2S fine-tuning (LoRA + Trainable Encoder) on device: {training_args_p2s.device}")
    trainer_p2s.train()
    
    print("P2S Fine-Tuning finished. Saving final model and LoRA adapter...")
    
    model.save_pretrained("./p2s_final_model_lora_trainable_encoder")
    tokenizer.save_pretrained("./p2s_final_model_lora_trainable_encoder")