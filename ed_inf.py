import torch
from transformers import AutoTokenizer, EncoderDecoderModel

# --- 1. 配置和词汇表定义 (必须与训练时一致) ---

# 提供的音素列表
PHONEMES_LIST = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "BLANK"
]

# 添加特殊标记和分隔符
SPECIAL_TOKENS = ['<pad>', '<sos>', '<eos>', '.', '|']
PHONEME_VOCAB = SPECIAL_TOKENS + PHONEMES_LIST
phoneme_to_id = {p: i for i, p in enumerate(PHONEME_VOCAB)}
PAD_ID_PHONEME = phoneme_to_id['<pad>']
BLANK_ID = phoneme_to_id['BLANK']
VOCAB_SIZE_PHONEME = len(PHONEME_VOCAB)

# 模型和分词器名称
PLM_MODEL_NAME = "gpt2"
MODEL_PATH = "./ed/p2s_checkpoints_lora/checkpoint-10000" # <--- 替换为您的最佳模型路径!

# --- 2. 模型和分词器加载 ---

# 加载 PLM Tokenizer（用于文本解码）
tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
if not tokenizer.pad_token:
    # GPT-2 默认没有 PAD token，将其设置为 EOS token (ID 50256)
    tokenizer.pad_token = tokenizer.eos_token 

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载训练好的 Encoder-Decoder 模型
try:
    model = EncoderDecoderModel.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check if MODEL_PATH is correct and the model finished training.")
    exit()

# 💡 关键修复步骤：显式设置 Decoder 的生成配置 (Generation Configuration)
# 这解决了 GPT-2 中 PAD, BOS, EOS ID 冲突的问题，确保生成逻辑正确。
# 这一步是前一次修复中已添加的。
model.config.decoder_start_token_id = tokenizer.bos_token_id 
model.config.pad_token_id = tokenizer.pad_token_id           
model.config.eos_token_id = tokenizer.eos_token_id           
print("✅ Decoder generation config updated.")


# --- 3. 推理函数 (主要修改在 model.generate 部分) ---

def phoneme_to_sentence(phoneme_sequence: str, model, tokenizer, device, max_length=50):
    """
    将音素序列转换为文本句子。
    """
    
    # --- 1. 预处理音素序列 (Encoder Input) ---
    
    phoneme_tokens = phoneme_sequence.split()
    cleaned_phoneme_tokens = [p for p in phoneme_tokens if p != 'BLANK']
    
    # 转换为 ID
    p_ids = [phoneme_to_id['<sos>']] + \
            [phoneme_to_id.get(p, PAD_ID_PHONEME) for p in cleaned_phoneme_tokens] + \
            [phoneme_to_id['<eos>']]
    
    # 转换为 PyTorch Tensor
    input_ids = torch.tensor([p_ids], dtype=torch.long).to(device)
    
    # 创建注意力掩码
    attention_mask = (input_ids != PAD_ID_PHONEME).long()

    # --- 2. 模型生成 (Generation) ---
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # 强制阻止 2-gram 或 3-gram 重复，有效减少 "TheThe" 或 "a a a" 的出现
            no_repeat_ngram_size=3, 
            # 其他参数现在应在 model.generation_config 中设置
        )
    
    # --- 3. 后处理和解码 ---
    
    generated_sentence = tokenizer.decode(
        generated_ids.squeeze().tolist(), 
        skip_special_tokens=True 
    )
    
    return generated_sentence.strip()


# --- 4. 示例推理 ---

example_phonemes = "DH AH | K R UH K AH D | M EY Z | F EY L D | T UW | F UW L | DH AH | M AW S |"

print("\n--- Starting Inference ---")
print(f"Input Phonemes: {example_phonemes}")

# 执行推理
generated_sentence = phoneme_to_sentence(
    example_phonemes, 
    model, 
    tokenizer, 
    device, 
    max_length=50
)

# 打印结果
print(f"\nGenerated Sentence: {generated_sentence}")
print("-" * 30)