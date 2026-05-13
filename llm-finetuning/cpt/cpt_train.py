import json
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)

from datasets import Dataset

# 설정
MODEL_NAME = "meta-llama/Llama-3.2-3B"
DATA_PATH = "data/chunked.jsonl"
OUTPUT_DIR = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/cpt"
LOG_DIR = "/workspace/LeeJeongmin-project/llm-finetuning/outputs/logs"
EPOCHS = 1
BATCH_SIZE = 4
LR = 2e-5

def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return Dataset.from_list(data)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )

dataset = load_data(DATA_PATH)
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    save_strategy="epoch",
    logging_steps=100,
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
print("CPT 완료")