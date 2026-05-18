from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from bert_score import score
from rouge_score import rouge_scorer

MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/sft"
EVAL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko_eval.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
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
    sentences = re.split(r'(?<=[다요])\s', answer)
    answer = ' '.join(sentences[:2]).strip()
    answer = re.split(r'\n[A-Za-z]', answer)[0].strip()
    answer = re.split(r'\s{2,}[A-Z][a-z]', answer)[0].strip()
    predictions.append(answer)
    references.append(item["output"])
    print(f"질문: {item['instruction'][:50]}...")
    print(f"생성: {answer[:100]}...")
    print()

P, R, F1 = score(predictions, references, lang="ko")
print(f"\nBERTScore - Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for pred, ref in zip(predictions, references):
    scores = scorer.score(ref, pred)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

print(f"ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
print(f"ROUGE-2: {sum(rouge2_scores)/len(rouge2_scores):.4f}")
print(f"ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}")