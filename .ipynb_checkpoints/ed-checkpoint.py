import pandas as pd
import torch
import numpy as np
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer, 
    BertConfig, 
    BertModel, 
    GPT2Config, 
    GPT2LMHeadModel, 
    EncoderDecoderConfig, 
    EncoderDecoderModel,
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)

# ==========================================
# 1. 全局配置 & 音素定义
# ==========================================

# 提供的音素列表
PHONEMES_LIST = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "BLANK"
]

SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '.', '|'] 
PHONEME_VOCAB = SPECIAL_TOKENS + PHONEMES_LIST
phoneme_to_id = {p: i for i, p in enumerate(PHONEME_VOCAB)}
id_to_phoneme = {i: p for p, i in phoneme_to_id.items()}

VOCAB_SIZE_PHONEME = len(PHONEME_VOCAB)
PAD_ID_PHONEME = phoneme_to_id['<pad>']
SOS_ID = phoneme_to_id['<sos>']
EOS_ID = phoneme_to_id['<eos>']

# 模型与训练超参数
PLM_MODEL_NAME = "gpt2"      # Decoder 基座
MAX_P_LEN = 128              # Encoder 输入最大长度 (音素)
MAX_S_LEN = 50               # Decoder 输出最大长度 (文本)
ENCODER_HIDDEN_SIZE = 768    
NUM_EPOCHS = 50              # 设置较高的 Epoch，依赖早停机制

# ==========================================
# 2. 数据处理函数
# ==========================================

# 加载 GPT-2 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
# 重要：将 pad_token 设置为 eos_token，避免引入新 token 导致维度不匹配
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(example):
    """
    数据预处理核心逻辑
    """
    # --- 1. 处理 Encoder 输入 (音素) ---
    phoneme_seq = str(example['phonemes']).split()
    # 过滤 BLANK
    cleaned_phoneme_seq = [p for p in phoneme_seq if p != 'BLANK']
    
    # 构建 ID 序列: <SOS> + Body + <EOS>
    p_ids = [SOS_ID] + \
            [phoneme_to_id.get(p, PAD_ID_PHONEME) for p in cleaned_phoneme_seq] + \
            [EOS_ID]
    
    # 截断
    if len(p_ids) > MAX_P_LEN:
        p_ids = p_ids[:MAX_P_LEN]
        p_ids[-1] = EOS_ID # 确保最后一位是 EOS
        
    # Encoder Attention Mask
    attention_mask_p = [1] * len(p_ids)
    
    # 手动 Padding (Encoder)
    padding_len = MAX_P_LEN - len(p_ids)
    if padding_len > 0:
        p_ids.extend([PAD_ID_PHONEME] * padding_len)
        attention_mask_p.extend([0] * padding_len)

    # --- 2. 处理 Decoder 目标 (文本) ---
    # 使用 Tokenizer 处理文本
    s_tokenized = tokenizer(
        example['sentence'], 
        max_length=MAX_S_LEN, 
        padding="max_length", 
        truncation=True
    )
    
    labels = s_tokenized['input_ids'].copy()
    
    # 将 Padding 部分的 Label 设为 -100 (Loss 计算忽略)
    labels = [
        l if l != tokenizer.pad_token_id else -100 
        for l in labels
    ]

    return {
        'input_ids': p_ids,                 # Encoder 输入
        'attention_mask': attention_mask_p, # Encoder Mask
        'labels': labels,                   # Decoder 目标 (用于计算 Loss 和生成)
    }

# ==========================================
# 3. 数据加载
# ==========================================

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t').dropna().reset_index(drop=True)
    return Dataset.from_pandas(df)
train_dataset_raw = load_data('train.tsv')
val_dataset_raw = load_data('val.tsv')

# 应用预处理
tokenized_train = train_dataset_raw.map(preprocess_function, remove_columns=['phonemes', 'sentence'])
tokenized_val = val_dataset_raw.map(preprocess_function, remove_columns=['phonemes', 'sentence'])

# ==========================================
# 4. 模型构建
# ==========================================

# 4.1 Encoder 配置 (BERT 结构，从头训练，减少层数以防过拟合)
encoder_config = BertConfig(
    vocab_size=VOCAB_SIZE_PHONEME, 
    hidden_size=ENCODER_HIDDEN_SIZE,         
    num_hidden_layers=4,        # 建议：对于小数据集，4层比6层更好收敛
    num_attention_heads=12,
    intermediate_size=ENCODER_HIDDEN_SIZE * 4,
    pad_token_id=PAD_ID_PHONEME
)
encoder = BertModel(config=encoder_config)
# 初始化 Embedding 层
encoder.embeddings.word_embeddings = torch.nn.Embedding(
    VOCAB_SIZE_PHONEME, ENCODER_HIDDEN_SIZE, padding_idx=PAD_ID_PHONEME
)

# 4.2 Decoder 配置 (GPT-2 预训练)
decoder_config = GPT2Config.from_pretrained(PLM_MODEL_NAME)
decoder_config.add_cross_attention = True
decoder_config.is_decoder = True 
decoder = GPT2LMHeadModel.from_pretrained(PLM_MODEL_NAME, config=decoder_config)

# 4.3 组合 Seq2Seq 模型
config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
model = EncoderDecoderModel(config=config, encoder=encoder, decoder=decoder)

# 4.4 关键 Token ID 设置
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# 4.5 生成参数设置 (解决复读机问题)
model.config.max_length = MAX_S_LEN
model.config.min_length = 2
model.config.no_repeat_ngram_size = 3   # 禁止生成重复的3-gram
model.config.early_stopping = True
model.config.length_penalty = 1.0
model.config.num_beams = 4              # Beam Search 宽度

# ==========================================
# 5. 评估指标设置
# ==========================================
try:
    wer_metric = evaluate.load("wer")
except Exception:
    print("Warning: 'evaluate' or 'jiwer' not found. Using simple accuracy.")
    wer_metric = None

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # 替换 -100 为 pad_id 以便解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # 解码
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 简单的清理
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # 计算 WER
    result = {}
    if wer_metric:
        wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"wer": wer}
    
    # --- 打印样例，直观监控效果 ---
    print("\n" + "="*30)
    print(f"Pred:  {decoded_preds[0]}")
    print(f"Label: {decoded_labels[0]}")
    print("="*30 + "\n")
    
    return result

# ==========================================
# 6. 训练器配置 (Seq2SeqTrainer)
# ==========================================

training_args = Seq2SeqTrainingArguments(
    output_dir="./p2s_checkpoints",
    num_train_epochs=NUM_EPOCHS,     # 设置大一点 (50)，靠 EarlyStopping 停
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    warmup_steps=100,                # 适当 warmup
    weight_decay=0.01,
    logging_steps=50,
    
    # 评估与保存策略
    eval_strategy="epoch",
    #eval_steps=100,
    save_strategy="epoch",
    #save_steps=100,
    
    # 预测生成配置
    predict_with_generate=True,      # 评估时执行生成
    generation_max_length=MAX_S_LEN,
    
    # 最佳模型加载
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", # 可以改为 "wer"
    greater_is_better=False,         # Loss 越小越好
    
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# 数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    model=model, 
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

# 实例化 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)] # 连续 10 次没提升就停止
)

# ==========================================
# 7. 开始训练
# ==========================================

if __name__ == "__main__":
    print(f"Starting training on device: {training_args.device}")
    trainer.train()
    print("Training finished. Saving final model...")
    trainer.save_model("./p2s_final_model")
    tokenizer.save_pretrained("./p2s_final_model")