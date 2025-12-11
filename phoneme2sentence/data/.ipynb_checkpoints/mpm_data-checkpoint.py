import os
import multiprocessing
from datasets import load_dataset
from g2p_en import G2p
from tqdm import tqdm
import time
import nltk
nltk.download('averaged_perceptron_tagger_eng')

# ================= Configuration Area =================
OUTPUT_FILE = "real_data_phonemes_large.tsv"
NUM_WORKERS = max(1, os.cpu_count() - 2)
BATCH_SIZE = 1000
# Set to None to process all samples
MAX_TOTAL_SAMPLES = 50000

# Target CMU Phonemes + BLANK
TARGET_PHONEMES = set([
    "AA", "AE", "AH", "AO", "AW", 
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", 
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH", "BLANK"
])
# ===========================================

g2p_model = None

def init_worker():
    """Worker process initialization"""
    global g2p_model
    # Initialize G2P model in each child process
    g2p_model = G2p()

def clean_phoneme(p):
    # Remove stress digits (e.g., 'AH0' -> 'AH')
    return ''.join([c for c in p if not c.isdigit()])

def process_batch(text_batch):
    global g2p_model
    results = []
    
    # Check if model is loaded
    if g2p_model is None:
        print("Error: G2P model is None in worker!")
        return []

    for text in text_batch:
        if not text or len(text) < 5:
            continue
            
        try:
            # text is already cleaned text
            raw_phonemes = g2p_model(text)
        except Exception as e:
            # Log error
            print(f"G2P Error: {e}")
            continue
            
        clean_seq = []
        for p in raw_phonemes:
            if p == ' ' or p == '': 
                continue
            
            # Check for valid phoneme
            if p.isalpha() or (len(p) > 1 and any(c.isdigit() for c in p)):
                p_final = clean_phoneme(p)
                if p_final in TARGET_PHONEMES:
                    clean_seq.append(p_final)
        
        # Only save if enough phonemes are extracted
        if len(clean_seq) >= 3:
            results.append(" ".join(clean_seq))
            
    return results

def data_generator(dataset, max_samples=None):
    batch = []
    count = 0
    print("Reading data stream from HuggingFace...")
    
    for sample in dataset:
        text_content = sample.get('text', "")
        
        # WikiText cleaning logic
        text_content = text_content.strip()
        if not text_content: 
            continue
        
        # Skip header lines " = = = Header = = = "
        if text_content.startswith("=") and text_content.endswith("="):
            continue
            
        # Simple newline removal
        text_content = text_content.replace('\n', ' ')
        
        # Length filter
        if len(text_content.split()) < 3:
            continue

        batch.append(text_content)
        count += 1
        
        if len(batch) == BATCH_SIZE:
            yield batch
            batch = []
            
        if max_samples and count >= max_samples:
            break
            
    if batch:
        yield batch

def ensure_nltk_resources():
    """Ensure all NLTK resources are downloaded"""
    print("Checking NLTK resources...")
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/cmudict')
        nltk.data.find('corpora/cmudict.zip')
    except LookupError:
        print("Downloading missing NLTK data (once only)...")
        nltk.download('averaged_perceptron_tagger')
        nltk.download('cmudict')
        nltk.download('averaged_perceptron_tagger_eng')

def main():
    # 1. Environment check
    ensure_nltk_resources()
    
    # 2. G2P smoke test
    print("Running G2P smoke test...")
    try:
        test_g2p = G2p()
        test_out = test_g2p("Hello world")
        print(f"G2P smoke test output: {test_out}")
        if not test_out:
            print("Error: G2P failed to generate output. Check your environment!")
            return
        del test_g2p # Release resource in main process
    except Exception as e:
        print(f"Error during G2P smoke test: {e}. Check g2p_en installation.")
        return

    # 3. Set multiprocessing start method
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"=== Starting Multiprocessing (WikiText) ===")
    print(f"Number of Workers: {NUM_WORKERS}")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    
    pool = multiprocessing.Pool(processes=NUM_WORKERS, initializer=init_worker)
    
    total_processed = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        # Wrap generator with tqdm
        iterator = pool.imap_unordered(process_batch, data_generator(dataset, MAX_TOTAL_SAMPLES))
        
        pbar = tqdm(iterator, unit="batch")
        for batch_results in pbar:
            if not batch_results:
                continue
                
            for line in batch_results:
                f.write(line + "\n")
                
            total_processed += len(batch_results)
            pbar.set_description(f"Saved: {total_processed} lines")

    pool.close()
    pool.join()
    
    print(f"\nProcessing complete!")
    print(f"File path: {OUTPUT_FILE}")
    
    # Final check
    if total_processed == 0 or os.path.getsize(OUTPUT_FILE) == 0:
        print("\n!!!!!!!! WARNING !!!!!!!!")
        print("Output file is empty! No data was successfully converted.")
        print("Please check the error log above.")
    else:
        print(f"Successfully generated data. You can now run train_mpm.py.")

if __name__ == "__main__":
    main()