import torch
from transformers import EncoderDecoderModel
import data_utils
import config
import os

def generate_sentence(model, phoneme_str):
    model.eval()
    device = model.device
    
    # 1. Tokenize Phonemes (Encoder Input)
    inputs = data_utils.phoneme_tokenizer.encode(phoneme_str, max_length=config.MAX_P_LEN)
    
    # Convert to Tensor
    input_ids = torch.tensor([inputs["input_ids"]]).to(device)
    attention_mask = torch.tensor([inputs["attention_mask"]]).to(device)
    
    # 2. Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.MAX_S_LEN,
            num_beams=5,                    # Increase Beam size
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,         # Enhanced penalty
            length_penalty=0.8,             # Key: Encourage shorter sentences
            early_stopping=True,
            decoder_start_token_id=model.config.decoder_start_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id
        )
    
    # 3. Decode
    decoded_text = data_utils.gpt_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded_text

def main():
    model_path = config.FINAL_MODEL_DIR
    
    if not os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        print(f"Error: Model file not found in {model_path}! Please ensure train_p2s_nopeft.py completed.")
        return

    print(f">>> Loading model from {model_path}...")
    model = EncoderDecoderModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(">>> Model loaded successfully.")

    # Test cases
    test_cases = [
        "AH | K AH P | AH V | SH UH G ER | M EY K S | S W IY T | F AH JH |",
        "P L EY S | AH | R OW Z B UH SH | N IH R | DH AH | P AO R CH | S T EH P S |",
        "B OW TH | L AO S T | DH EH R | L IH V Z | IH N | DH AH | R EY JH IH NG | S T AO R M |"
    ]

    print("\n>>> Testing with Length Penalty (0.8) and Repetition Penalty (1.5):")
    for phones in test_cases:
        print(f"\nInput:  {phones}")
        output = generate_sentence(model, phones)
        print(f"Output: {output}")

if __name__ == "__main__":
    main()