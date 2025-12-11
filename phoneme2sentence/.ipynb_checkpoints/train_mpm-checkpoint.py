# train_mpm.py
import os
import torch
from transformers import BertConfig, BertForMaskedLM, Trainer, TrainingArguments, set_seed
import config
import data_utils

def run_mpm():
    set_seed(config.SEED)
    print(">>> Starting MPM Pre-training...")

    # 1. Load Data
    train_ds, val_ds = data_utils.load_and_process_data(mode="mpm")
    if train_ds is None:
        print("Data not found. Exiting.")
        return

    # 2. Config & Model
    # Use config values for consistency with P2S loading
    model_config = BertConfig(
        vocab_size=config.VOCAB_SIZE_PHONEME,
        hidden_size=config.ENCODER_HIDDEN_SIZE,
        num_hidden_layers=config.ENCODER_NUM_LAYERS,
        num_attention_heads=config.ENCODER_ATTN_HEADS,
        intermediate_size=config.ENCODER_HIDDEN_SIZE * 4,
        max_position_embeddings=config.MAX_P_LEN,
        pad_token_id=config.PAD_ID_PHONEME
    )
    
    model = BertForMaskedLM(config=model_config)
    print(f"Model Created. Vocab Size: {model_config.vocab_size}")

    # 3. Collator
    collator = data_utils.PhonemeMaskingCollator(mask_prob=0.15)

    # 4. Training Args
    args = TrainingArguments(
        output_dir=config.MPM_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=config.EPOCHS_MPM,
        per_device_train_batch_size=config.BATCH_SIZE_MPM,
        learning_rate=config.LR_MPM,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator
    )

    trainer.train()

    # 5. Save ONLY the Encoder (bert) part for P2S
    # Save 'model.bert' to be loaded as 'BertModel'
    print(f"Saving encoder to {config.BEST_ENCODER_PATH}...")
    model.bert.save_pretrained(config.BEST_ENCODER_PATH)
    
    # Also save the tokenizer vocab
    data_utils.phoneme_tokenizer.save_pretrained(config.BEST_ENCODER_PATH)
    print("Done.")

if __name__ == "__main__":
    run_mpm()