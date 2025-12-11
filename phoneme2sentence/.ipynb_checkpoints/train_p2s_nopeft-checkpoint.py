# train_p2s_nopeft.py
import os
import torch
import numpy as np
from transformers import (
    BertModel, GPT2Config, GPT2LMHeadModel,
    EncoderDecoderModel,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, set_seed, EarlyStoppingCallback
)
import config
import data_utils

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    decoded_preds = data_utils.gpt_tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, data_utils.gpt_tokenizer.pad_token_id)
    decoded_labels = data_utils.gpt_tokenizer.batch_decode(labels, skip_special_tokens=True)

    print("\n--- Val Sample ---")
    # Print the first 3 samples
    for i in range(min(3, len(decoded_preds))):
        print(f"Ref : {decoded_labels[i]}")
        print(f"Pred: {decoded_preds[i]}")
        print("-" * 20)
        
    return {"val_example_gen": 1.0}

def run_p2s():
    set_seed(config.SEED)
    print(">>> Starting P2S Full Fine-tuning (No LoRA)...")

    # 1. Data
    train_ds, val_ds = data_utils.load_and_process_data(mode="p2s")

    # 2. Load Models [CRITICAL FIX HERE]
    print(f"Loading Encoder from: {config.BEST_ENCODER_PATH}")
    encoder = BertModel.from_pretrained(config.BEST_ENCODER_PATH)
    
    print(f"Loading Decoder: {config.PLM_MODEL_NAME}")
    # -------------------------------------------------------
    # CRITICAL FIX: Configure Config first, then load model
    # This ensures Cross-Attention layers are created
    # -------------------------------------------------------
    decoder_config = GPT2Config.from_pretrained(config.PLM_MODEL_NAME)
    decoder_config.add_cross_attention = True
    decoder_config.is_decoder = True
    decoder_config.pad_token_id = data_utils.gpt_tokenizer.pad_token_id
    decoder_config.eos_token_id = data_utils.gpt_tokenizer.eos_token_id
    decoder_config.bos_token_id = data_utils.gpt_tokenizer.eos_token_id

    # Use ignore_mismatched_sizes=True
    decoder = GPT2LMHeadModel.from_pretrained(
        config.PLM_MODEL_NAME, 
        config=decoder_config,
        ignore_mismatched_sizes=True 
    )

    # 3. Create Seq2Seq Model
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    
    # Set Model Config
    model.config.decoder_start_token_id = decoder.config.bos_token_id
    model.config.eos_token_id = decoder.config.eos_token_id
    model.config.pad_token_id = decoder.config.pad_token_id
    model.config.vocab_size = decoder.config.vocab_size
    model.config.max_length = config.MAX_S_LEN
    model.config.num_beams = 4
    
    # Repetition prevention
    model.config.no_repeat_ngram_size = 3
    model.config.repetition_penalty = 1.2

    print(f"Model Parameters: {model.num_parameters() / 1e6:.2f}M")

    # 4. Collator
    collator = DataCollatorForSeq2Seq(
        tokenizer=data_utils.gpt_tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    # 5. Trainer
    # Training arguments optimized for full fine-tuning
    args = Seq2SeqTrainingArguments(
        output_dir=config.P2S_OUTPUT_DIR,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2, 
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        learning_rate=config.LR_P2S, 
        num_train_epochs=config.EPOCHS_P2S,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        generation_config=model.generation_config
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    trainer.train()

    # 6. Save
    print(f"Saving Final Model to {config.FINAL_MODEL_DIR}")
    # Standard model, direct save
    model.save_pretrained(config.FINAL_MODEL_DIR)
    data_utils.gpt_tokenizer.save_pretrained(config.FINAL_MODEL_DIR)
    data_utils.phoneme_tokenizer.save_pretrained(config.FINAL_MODEL_DIR)

if __name__ == "__main__":
    run_p2s()