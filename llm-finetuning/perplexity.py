import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/sft"
EVAL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/data/eval_domain.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
model.eval()

total_loss = 0
total_tokens = 0

with open(EVAL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = f"### 질문:\n{item['instruction']}\n\n### 답변:\n{item['output']}"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
        total_tokens += inputs["input_ids"].shape[1]

perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
print(f"Perplexity: {perplexity:.4f}")