from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ===== 모드 설정 =====
MODE = "sft"  # "sft" 또는 "qlora"
# ====================

SFT_MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/sft"
QLORA_MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_final"

model_path = SFT_MODEL_PATH if MODE == "sft" else QLORA_MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

questions = [
    "### 질문:\n대한민국의 수도는 어디인가요?\n\n### 답변:",
    "### 질문:\ntumor cell에서 apoptosis가 일어나는 메커니즘은 무엇인가요?\n\n### 답변:",
    "### 질문:\ncancer cell의 morphology가 정상 세포와 다른 점은 무엇인가요?\n\n### 답변:"
]

for q in questions:
    print(f"\n질문: {q}")
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        repetition_penalty=1.5,
        do_sample=True,
        temperature=0.1,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"답변: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")