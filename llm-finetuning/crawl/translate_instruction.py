import json
import os
import time
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

with open('/home/jml3227/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko_1400.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# 계정 1: data[:500]      → pubmedqa_ko_fixed_1.jsonl
# 계정 2: data[500:1000]  → pubmedqa_ko_fixed_2.jsonl
# 계정 3: data[1000:]     → pubmedqa_ko_fixed_3.jsonl
data = data[975:]
output_path = '/home/jml3227/LeeJeongmin-project/llm-finetuning/data/pubmedqa_ko_fixed_3.jsonl'

prompt_template = """아래는 영어 질문과 한국어 답변 쌍입니다.
instruction은 한국어로 번역하고, output은 핵심 내용만 1~2문장으로 요약해서 한국어로 작성해주세요.

규칙:
- instruction과 output 모두 한국어로 작성
- 번역 가능한 의학 용어는 한국어로 번역 (예: tumor → 종양, cell → 세포)
- 고유 단백질명, 유전자명, 약물명은 영어 유지 (예: Bcl-2, EGFR, mTOR)
- output은 반드시 1~2문장으로 요약
- output은 친절한 대화체로 작성 (예: "~입니다", "~합니다" 형식. "~시사합니다", "~보증합니다" 같은 논문체 표현은 피할 것)
- 문장은 반드시 한국어로 끝낼 것
- JSON 배열 형식 그대로 유지
- 다른 설명 없이 JSON 배열만 출력

{batch}"""

results = []
batch_size = 25

for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    batch_str = json.dumps(batch, ensure_ascii=False, indent=2)
    print(f"요청 중: {i+1}~{min(i+batch_size, len(data))}번째")

    for attempt in range(5):
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
            break
        except Exception as e:
            print(f"  실패 (시도 {attempt+1}): {e}")
            if attempt < 4:
                time.sleep(30)
            else:
                print(f"  최대 재시도 초과, 건너뜀")

with open(output_path, 'w') as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"총 {len(results)}개 저장 완료")