# inspect_phoneme_data.py
import os
from collections import Counter, defaultdict
import math
import numpy as np # Move import up

FILE = "real_data_phonemes_large.tsv"
if not os.path.exists(FILE):
    raise SystemExit(f"{FILE} not found")

lengths = []
tok_counter = Counter()
line_counter = 0
too_short = 0
repeat_lines = 0
max_repeat_threshold = 0.6  # Sequences with one token >60% are considered repetitive

repeat_examples = []
short_examples = []

with open(FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        s = line.strip()
        if not s:
            continue
        line_counter += 1
        toks = s.split()
        lengths.append(len(toks))
        tok_counter.update(toks)
        if len(toks) < 3:
            too_short += 1
            if len(short_examples) < 10:
                short_examples.append(s)
        
        # Check per-line dominance
        perline = Counter(toks)
        top_tok, top_cnt = perline.most_common(1)[0]
        if top_cnt / len(toks) >= max_repeat_threshold:
            repeat_lines += 1
            if len(repeat_examples) < 10:
                repeat_examples.append(s)

# Compute stats
lengths = np.array(lengths) if lengths else np.array([0])
print("Total lines:", line_counter)
print("Sequence length: min, median, mean, 90pct, max =", lengths.min(), np.median(lengths), lengths.mean(), np.percentile(lengths,90), lengths.max())
print("Lines with <3 tokens:", too_short)
print("Lines with high token repetition (>60% same token):", repeat_lines)
print("Vocab size:", len(tok_counter))
print("Top 20 tokens:", tok_counter.most_common(20))

# Print some examples
print("\nShort examples (≤3 tokens):")
for s in short_examples[:10]:
    print(" ", s)
print("\nRepeat examples (dominant token):")
for s in repeat_examples[:10]:
    print(" ", s)