import json

files = [
    '/home/jml3227/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko_final.jsonl',
    '/home/jml3227/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko_fixed_4.jsonl',
    '/home/jml3227/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko_fixed_5.jsonl',
    '/home/jml3227/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko_fixed_6.jsonl',
    '/home/jml3227/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko_fixed_7.jsonl',
]

results = []
for f in files:
    with open(f, 'r') as fp:
        for line in fp:
            results.append(json.loads(line))

output_path = '/home/jml3227/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko_2975.jsonl'
with open(output_path, 'w') as fp:
    for item in results:
        fp.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"총 {len(results)}개 저장 완료")