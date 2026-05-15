from datasets import load_dataset

dataset = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

keywords_required = ['cell']
keywords_any = ['tumor', 'pathology', 'histology', 'morphology', 'cancer']

def filter_fn(example):
    text = (example['question'] + ' ' + example['long_answer']).lower()
    has_cell = 'cell' in text
    has_any = any(k in text for k in keywords_any)
    return has_cell and has_any

filtered = dataset.filter(filter_fn)
filtered = filtered.shuffle(seed=42).select(range(500))

def to_qa(example):
    return {
        "instruction": example['question'],
        "output": example['long_answer']
    }

converted = filtered.map(to_qa, remove_columns=filtered.column_names)
converted.to_json('/workspace/LeeJeongmin-project/llm-finetuning/data/pubmedqa_filtered.jsonl')

print(f"{len(converted)}개")