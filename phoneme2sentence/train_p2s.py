# train_p2s.py
import os
import torch
import numpy as np
from transformers import (
    BertModel, GPT2Config, GPT2LMHeadModel,
    EncoderDecoderConfig, EncoderDecoderModel,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, set_seed, EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import config
import data_utils

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Replace -100 with pad token id for decoding
    decoded_preds = data_utils.gpt_tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, data_utils.gpt_tokenizer.pad_token_id)
    decoded_labels = data_utils.gpt_tokenizer.batch_decode(labels, skip_special_tokens=True)

    print("\n--- Val Sample ---")
    if len(decoded_labels) > 0:
        print(f"Ref : {decoded_labels[0]}")
        print(f"Pred: {decoded_preds[0]}")
    return {"val_example_gen": 1.0}

def run_p2s():
    set_seed(config.SEED)
    print(">>> Starting P2S Fine-tuning...")

    # 1. Data
    train_ds, val_ds = data_utils.load_and_process_data(mode="p2s")

    # 2. Load Models
    print(f"Loading Encoder from: {config.BEST_ENCODER_PATH}")
    encoder = BertModel.from_pretrained(config.BEST_ENCODER_PATH)
    
    print(f"Loading Decoder Config: {config.PLM_MODEL_NAME}")
    # [CRITICAL FIX]: Configure BEFORE loading model
    decoder_config = GPT2Config.from_pretrained(config.PLM_MODEL_NAME)
    decoder_config.add_cross_attention = True
    decoder_config.is_decoder = True
    
    # Sync token IDs in config before loading model
    decoder_config.pad_token_id = data_utils.gpt_tokenizer.pad_token_id
    decoder_config.eos_token_id = data_utils.gpt_tokenizer.eos_token_id
    decoder_config.bos_token_id = data_utils.gpt_tokenizer.eos_token_id

    print("Loading Decoder Model with Cross-Attention...")
    # Warning is EXPECTED and CORRECT for CrossAttn
    decoder = GPT2LMHeadModel.from_pretrained(
        config.PLM_MODEL_NAME, 
        config=decoder_config,
        ignore_mismatched_sizes=True 
    )

    # 3. Create Seq2Seq Model
    model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
    
    # Explicitly set model config parameters for generation
    model.config.decoder_start_token_id = decoder_config.bos_token_id
    model.config.eos_token_id = decoder_config.eos_token_id
    model.config.pad_token_id = decoder_config.pad_token_id
    model.config.vocab_size = decoder_config.vocab_size
    model.config.max_length = config.MAX_S_LEN
    model.config.num_beams = 4

    # 4. LoRA Injection
    print("Injecting LoRA...")
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=["c_attn"], # Applies to self-attn and cross-attn projections
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM 
    )
    # Apply LoRA to the decoder
    model.decoder = get_peft_model(model.decoder, lora_config)
    model.decoder.print_trainable_parameters()

    # 5. Collator
    collator = DataCollatorForSeq2Seq(
        tokenizer=data_utils.gpt_tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )

    # 6. Trainer
    args = Seq2SeqTrainingArguments(
        output_dir=config.P2S_OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE_P2S,
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        learning_rate=config.LR_P2S,
        num_train_epochs=config.EPOCHS_P2S,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,    # Frequent eval
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none"
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

    # 7. Save
    print(f"Saving Final Model to {config.FINAL_MODEL_DIR}")

    from peft import PeftModel
    if isinstance(model.decoder, PeftModel):
        print("Merging LoRA adapters into base model for safe saving...")
        model.decoder = model.decoder.merge_and_unload()

    model.save_pretrained(config.FINAL_MODEL_DIR)
    
    data_utils.gpt_tokenizer.save_pretrained(config.FINAL_MODEL_DIR)
    data_utils.phoneme_tokenizer.save_pretrained(config.FINAL_MODEL_DIR)

if __name__ == "__main__":
    run_p2s()