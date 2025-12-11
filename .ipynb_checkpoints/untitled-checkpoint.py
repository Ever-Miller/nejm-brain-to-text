import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW

# ==========================================================
# Dataset
# ==========================================================
class PhonemeTSVDataset(Dataset):
    def __init__(self, tsv_path):
        self.data = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                ph, sent = line.strip().split("\t")
                phs = [x for x in ph.split("|") if x and x != "BLANK"]
                prompt = (
                    "### Instruction:\nConvert phonemes to a sentence.\n\n"
                    "### Phonemes:\n" + " ".join(phs) + "\n\n### Output:\n"
                )
                self.data.append((prompt, sent))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, target = self.data[idx]
        text = prompt + target
        return text


def collate_fn(batch, tokenizer):
    return tokenizer(batch,
                     padding=True,
                     truncation=True,
                     max_length=512,
                     return_tensors="pt")


# ==========================================================
# Training
# ==========================================================
def train():
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    device = "cuda"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model to GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    ds = PhonemeTSVDataset("train.tsv")
    dl = DataLoader(ds, batch_size=1, shuffle=True,
                    collate_fn=lambda x: collate_fn(x, tokenizer))

    optim = AdamW(model.parameters(), lr=5e-5)

    model.train()

    for epoch in range(3):
        print(f"===== Epoch {epoch+1} =====")
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss

            loss.backward()
            optim.step()
            optim.zero_grad()

            print(f"Loss: {loss.item():.4f}")

    model.save_pretrained("phi3_finetuned")
    tokenizer.save_pretrained("phi3_finetuned")
    print("Training finished!")


if __name__ == "__main__":
    train()
