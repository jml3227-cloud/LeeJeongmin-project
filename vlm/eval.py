from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch
import json
from PIL import Image
from bert_score import score
from rouge_score import rouge_scorer

MODEL_PATH = "/workspace/llava-onevision-qwen2-7b-ov-hf"
ADAPTER_PATH = "/workspace/LeeJeongmin-project/vlm/outputs/checkpoints"
EVAL_PATH = "/workspace/LeeJeongmin-project/vlm/data/qa_eval.json"

# 모델 로드
processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# 데이터 로드
with open(EVAL_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

data = data[:10]

predictions = []
references = []

for item in data:
    image_path = item["image"]
    image = Image.open(image_path).convert("RGB")

    # 첫 번째 human 질문, 첫 번째 gpt 답변만 사용
    user_text = item["conversations"][0]["value"].replace("<image>", "").strip()
    reference = item["conversations"][1]["value"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_text}
        ]
    }]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated = processor.decode(outputs[0], skip_special_tokens=True)
    answer = generated.split("assistant\n")[-1].strip()

    predictions.append(answer)
    references.append(reference)
    print(f"이미지: {image_path.split('/')[-1]}")
    print(f"생성: {answer[:100]}...")
    print()

# BERTScore
P, R, F1 = score(predictions, references, lang="ko")
print(f"\nBERTScore - Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")

# ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

for pred, ref in zip(predictions, references):
    scores = scorer.score(ref, pred)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

print(f"ROUGE-1: {sum(rouge1_scores)/len(rouge1_scores):.4f}")
print(f"ROUGE-2: {sum(rouge2_scores)/len(rouge2_scores):.4f}")
print(f"ROUGE-L: {sum(rougeL_scores)/len(rougeL_scores):.4f}")