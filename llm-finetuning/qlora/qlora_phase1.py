import os
import gc 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset

BASE_MODEL="meta-llama/Llama-3.2-3B"
PHASE1_OUTPUT = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_phase1"
MERGED_OUTPUT = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_merged"
LOG_DIR = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_phase1"
EPOCHS = 2
BATCH_SIZE = 4
LR = 2e-4

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

def format_alpaca(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    if input_text:
        instruction = f"{instruction}\n{input_text}"
    text = f"### 질문:\n{instruction}\n\n### 답변:\n{example['output']}"
    return {"text": text}

alpaca = load_dataset("tatsu-lab/alpaca", split="train")
alpaca = alpaca.filter(lambda x: 'http' not in x['output'] and len(x['output']) > 0)
alpaca = alpaca.filter(lambda x: len(x['output']) < 300)
alpaca = alpaca.map(format_alpaca, remove_columns=alpaca.column_names)
alpaca = alpaca.shuffle(seed=42).select(range(200))

os.makedirs(PHASE1_OUTPUT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

args = TrainingArguments(
    output_dir=PHASE1_OUTPUT,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    save_strategy="no",
    logging_steps=10,
    bf16=True,
    tf32=True,
    lr_scheduler_type="constant",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    report_to="tensorboard",
    logging_dir=LOG_DIR,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=alpaca,
    peft_config=lora_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    dataset_text_field="text"
)

trainer.train()
trainer.save_model(PHASE1_OUTPUT)
tokenizer.save_pretrained(PHASE1_OUTPUT)
print("Phase 1 완료")

del model
del trainer
gc.collect()
torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(
    PHASE1_OUTPUT,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

merged = model.merge_and_unload()
merged.save_pretrained(MERGED_OUTPUT)
tokenizer.save_pretrained(MERGED_OUTPUT)
print("머지 완료")