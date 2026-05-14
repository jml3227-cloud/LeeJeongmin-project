import json
import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

categories = [
    "H&E staining 색깔과 의미",
    "세포핵 형태학",
    "박테리아 형태 분류",
]

prompt_template = """당신은 세포 형태학과 병리학 전문가입니다.
아래 주제에 대한 질문-답변 쌍 10개를 생성해주세요.

주제: {category}

규칙:
- 전문 용어는 영어로 사용
- 설명은 한국어로 작성
- sft_domain.jsonl에 있는 질문과 겹치지 않는 새로운 질문으로

반드시 아래 JSON 형식으로만 응답하세요:
[
  {{"instruction": "질문", "output": "답변"}},
  ...
]"""

os.makedirs("data", exist_ok=True)

total = 0

with open("data/eval_domain.jsonl", "w", encoding="utf-8") as f:
    for category in categories:
        print(f"생성 중: {category}")
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_template.format(category=category)
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            items = json.loads(text)
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            total += len(items)
            print(f"  {len(items)}개 생성 완료 (누적: {total}개)")
        except Exception as e:
            print(f"  실패: {e}")
            continue

print(f"\n총 {total}개 저장 완료")