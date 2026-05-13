from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ===== 모드 설정 =====
MODE = "sft"  # "cpt" 또는 "sft"
# ====================

CPT_MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/cpt"
SFT_MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/sft"

model_name = CPT_MODEL_PATH if MODE == "cpt" else SFT_MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

questions_cpt = [
    "대한민국의 수도는 어디인가요?",
    "H&E 염색에서 보라색은 무엇을 염색한 건가요?"
]

questions_sft = [
    "### 질문:\n대한민국의 수도는 어디인가요?\n\n### 답변:",
    "### 질문:\nH&E 염색에서 보라색은 무엇을 염색한 건가요?\n\n### 답변:"
]

questions = questions_cpt if MODE == "cpt" else questions_sft

for q in questions:
    print(f"\n질문: {q}")
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.7,
    )
    print(f"답변: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")