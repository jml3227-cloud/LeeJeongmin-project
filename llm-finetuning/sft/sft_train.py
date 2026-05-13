import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from trl import DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset, concatenate_datasets

CPT_MODEL_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/cpt"
DATA_PATH = "/workspace/LeeJeongmin-project/llm-finetuning/data/sft_domain.jsonl"
OUTPUT_DIR = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/sft"
LOG_DIR = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/logs"
EPOCHS = 3
BATCH_SIZE = 4
LR = 1e-5

tokenizer = AutoTokenizer.from_pretrained(CPT_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(CPT_MODEL_PATH)

def load_domain_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append({"instruction": item["instruction"], "output": item["output"]})
    return Dataset.from_list(data)

def format_alpaca_en(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    if input_text:
        instruction = f"{instruction}\n{input_text}"
    return {"instruction": instruction, "output": example["output"]}

def format_alpaca_ko(example):
    return {"instruction": example["instruction"], "output": example["output"]}

def format_prompt(example):
    text = f"### 질문:\n{example['instruction']}\n\n### 답변:\n{example['output']}"
    result = tokenizer(
        text,
        truncation=True,
        max_length=2048,
    )
    return result

# 도메인 데이터
domain_dataset = load_domain_data(DATA_PATH)

# Alpaca
alpaca_en = load_dataset("tatsu-lab/alpaca", split="train")
alpaca_en = alpaca_en.filter(lambda x: len(x["output"]) > 0)
alpaca_en = alpaca_en.map(format_alpaca_en, remove_columns=alpaca_en.column_names)
alpaca_en = alpaca_en.shuffle(seed=42).select(range(200))

# Alpaca-ko
# alpaca_ko = load_dataset("beomi/KoAlpaca-v1.1a", split="train")
# alpaca_ko = alpaca_ko.map(format_alpaca_ko, remove_columns=alpaca_ko.column_names)
# alpaca_ko = alpaca_ko.shuffle(seed=42).select(range(200))

# concat
dataset = concatenate_datasets([domain_dataset, alpaca_en])
dataset = dataset.shuffle(seed=42)
dataset = dataset.map(format_prompt, remove_columns=["instruction", "output"])

response_template_ids = tokenizer.encode("### 답변:\n", add_special_tokens=False)
data_collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    save_strategy="no",
    logging_steps=10,
    bf16=True,
    report_to="tensorboard",
    logging_dir=LOG_DIR
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("SFT 완료")