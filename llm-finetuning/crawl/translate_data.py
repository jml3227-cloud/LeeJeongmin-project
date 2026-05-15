import json
import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

with open('/home/jml3227/LeeJeongmin-project/llm-finetuning/data/pubmedqa_filtered.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

prompt_template = """아래 영어 질문-답변 쌍을 한국어로 번역해주세요.
규칙:
- 전문 용어(hematoxylin, eosin, apoptosis, tumor 등 의학/생물학 용어)는 영어 그대로 유지
- 나머지 설명은 자연스러운 한국어로 번역
- JSON 형식 그대로 유지
- 다른 설명 없이 JSON만 출력

{batch}"""

os.makedirs('data', exist_ok=True)

results = []
batch_size = 25

for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    batch_str = json.dumps(batch, ensure_ascii=False, indent=2)
    
    print(f"요청 중: {i+1}~{min(i+batch_size, len(data))}번째")
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_template.format(batch=batch_str)
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        items = json.loads(text)
        results.extend(items)
        print(f"  완료 ({len(results)}개 누적)")
    except Exception as e:
        print(f"  실패: {e}")
        continue

with open('/home/jml3227/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko.jsonl', 'w') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"총 {len(results)}개 저장 완료")