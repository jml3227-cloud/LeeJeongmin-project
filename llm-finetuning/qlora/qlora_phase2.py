import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets

BASE_MODEL = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_merged"
PHASE2_OUTPUT = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_phase2"
FINAL_OUTPUT = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_final"
LOG_DIR = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/logs"
DOMAIN_DATA_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko.jsonl"
EPOCHS = 2
BATCH_SIZE = 4
LR = 1e-5

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
    r=64,
    lora_alpha=128,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

def format_prompt(example):
    text = f"### 질문:\n{example['instruction']}\n\n### 답변:\n{example['output']}"
    return {"text": text}

def format_alpaca(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    if input_text:
        instruction = f"{instruction}\n{input_text}"
    text = f"### 질문:\n{instruction}\n\n### 답변:\n{example['output']}"
    return {"text": text}

domain_dataset = load_dataset("json", data_files=DOMAIN_DATA_PATH, split="train")
domain_dataset = domain_dataset.map(format_prompt, remove_columns=domain_dataset.column_names)

alpaca = load_dataset("tatsu-lab/alpaca", split="train")
alpaca = alpaca.filter(lambda x: 'http' not in x['output'] and len(x['output']) > 0)
alpaca = alpaca.map(format_alpaca, remove_columns=alpaca.column_names)
alpaca = alpaca.shuffle(seed=99).select(range(200))

dataset = concatenate_datasets([domain_dataset, alpaca]).shuffle(seed=42)

os.makedirs(PHASE2_OUTPUT, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

args = TrainingArguments(
    output_dir=PHASE2_OUTPUT,
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
    train_dataset=dataset,
    peft_config=lora_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    dataset_text_field="text",
)

trainer.train()
trainer.save_model(PHASE2_OUTPUT)
tokenizer.save_pretrained(PHASE2_OUTPUT)
print("Phase2 완료")

gc.collect()
torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(
    PHASE2_OUTPUT,
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

merged = model.merge_and_unload()
merged.save_pretrained(FINAL_OUTPUT)
tokenizer.save_pretrained(FINAL_OUTPUT)
print("최종 머지 완료")