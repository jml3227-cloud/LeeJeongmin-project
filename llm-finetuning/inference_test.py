from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# ===== 모드 설정 =====
MODE = "qlora"  # "base", "sft", "qlora_phase1", "qlora"
# ====================

BASE_MODEL_PATH = "meta-llama/Llama-3.2-3B"
QLORA_PHASE1_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_merged"
SFT_MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/sft"
QLORA_MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_final"

model_path = {
    "base": BASE_MODEL_PATH,
    "sft": SFT_MODEL_PATH,
    "qlora_phase1": QLORA_PHASE1_PATH,
    "qlora": QLORA_MODEL_PATH,
}[MODE]

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

questions = [
    "대한민국의 수도는 어디인가요?",
    "Bcl-2 과발현이 종양 세포의 방사선 저항성과 관련이 있나요?",
]

for q in questions:
    print(f"\n질문: {q}")
    q_with_prompt = f"간결하게 1~2문장으로 답하세요.\n\n### 질문:\n{q}\n\n### 답변:"
    inputs = tokenizer(q_with_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        repetition_penalty=1.5,
        do_sample=True,
        temperature=0.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split('### 답변:')[-1].strip()
    sentences = re.split(r'(?<=[다요])\s', answer)
    answer = ' '.join(sentences[:2]).strip()
    answer = re.split(r'\n[A-Za-z]', answer)[0].strip()
    answer = re.split(r'\s{2,}[A-Z][a-z]', answer)[0].strip()
    answer = re.sub(r'\(https?://\S+\)', '', answer).strip()
    answer = re.sub(r'https?://\S+', '', answer).strip()
    print(f"답변: {answer}")