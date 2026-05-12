import json
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

CHUNK_SIZE = 1024
OVERLAP = 0

def chunk_text(text):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += CHUNK_SIZE - OVERLAP
    return chunks

os.makedirs("data", exist_ok=True)

input_files = [
    "data/pmc_raw.jsonl",
    "data/wiki_raw.jsonl"
]

results = []

for input_file in input_files:
    if not os.path.exists(input_file):
        print(f"파일 없음: {input_file}")
        continue
    print(f"처리 중: {input_file}")
    with open(input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            if not text:
                continue
            chunks = chunk_text(text)
            for chunk in chunks:
                results.append({"text": chunk})

with open("data/chunked.jsonl", "w") as f:
    for item in results:
        f.write(json.dumps(item) + "\n")

print(f"\n 총 {len(results)}개 청크 저장 완료")