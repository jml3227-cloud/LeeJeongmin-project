from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from bert_score import score

MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/sft"
EVAL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/data/eval_domain.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

data = []
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

predictions = []
references = []

for item in data:
    prompt = f"### 질문:\n{item['instruction']}\n\n### 답변:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        repetition_penalty=1.5,
        do_sample=True,
        temperature=0.5,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated.split("### 답변:")[-1].strip()
    predictions.append(answer)
    references.append(item["output"])
    print(f"질문: {item['instruction'][:50]}...")
    print(f"생성: {answer[:100]}...")
    print()

P, R, F1 = score(predictions, references, lang="ko")
print(f"\nBERTScore - Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")