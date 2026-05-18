from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# ===== 모드 설정 =====
MODE = "qlora"  # "sft" 또는 "qlora"
# ====================

SFT_MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/sft"
QLORA_MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_final"

model_path = SFT_MODEL_PATH if MODE == "sft" else QLORA_MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

questions = [
    "### 질문:\n대한민국의 수도는 어디인가요?\n\n### 답변:",
    "### 질문:\ncold tumor를 hot tumor로 전환하는 방법은 무엇인가요?\n\n### 답변:",
    "### 질문:\ntumor microenvironment에서 면역 억제가 일어나는 메커니즘은 무엇인가요?\n\n### 답변:"
]

for q in questions:
    print(f"\n질문: {q}")
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        repetition_penalty=1.5,
        do_sample=True,
        temperature=0.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        min_new_tokens=20,
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split('###답변:')[-1].strip()
    answer = re.split(r'\n[A-Za-z]', answer)[0].strip()
    print(f"답변: {answer}")