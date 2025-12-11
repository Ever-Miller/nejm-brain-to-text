import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model

# ============================================================
# 1. Load Meta Llama 3.1 8B
# ============================================================
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    padding_side="left",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    trust_remote_code=True
)

# ============================================================
# 2. Add phoneme tokens
# ============================================================
phoneme_tokens = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "BLANK"
]

added = tokenizer.add_tokens(phoneme_tokens)
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
print(f"Added {added} phoneme tokens.")

# ============================================================
# 3. Build LoRA
# ============================================================
lora_config = LoraConfig(
    r=64,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("LoRA applied.")

# ============================================================
# 4. Load TSV
# ============================================================
ds = load_dataset("csv", data_files={
    "train": "train.tsv",
    "val": "val.tsv"
}, delimiter="\t")


# ============================================================
# 5. Formatting
# ============================================================
def format_sample(e):
    # phoneme 用 " | " 分割，并删除 BLANK
    phs = [x.strip() for x in e["phonemes"].split("|")]
    phs = [p for p in phs if p != "BLANK" and p != ""]

    phoneme_str = " ".join(phs)

    prompt = (
        f"### Instruction:\n"
        f"Convert the following phoneme sequence into an English sentence.\n\n"
        f"### Phonemes:\n{phoneme_str}\n\n"
        f"### Output:\n"
    )

    text = prompt + e["sentence"]

    tokenized = tokenizer(
        text,
        max_length=512,
        truncation=True
    )
    return tokenized

ds = ds.map(format_sample)

# ============================================================
# 6. Train
# ============================================================
args = TrainingArguments(
    output_dir="qwen-ph2text",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["val"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()

model.save_pretrained("qwen-ph2text-lora")
tokenizer.save_pretrained("qwen-ph2text-lora")
