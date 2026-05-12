import anthropic
import json
import os

client = anthropic.Anthropic()

categories = [
    "H&E staining 색깔과 의미",
    "세포핵 형태학",
    "박테리아 형태 분류",
    "병리학적 소견",
    "현미경 이미지 해석",
]

prompt_template = """당신은 세포 형태학과 병리학 전문가입니다.
아래 주제에 대한 질문-답변 쌍 10개를 생성해주세요.

주제: {category}

규칙:
- 전문 용어는 영어로 사용 (예: H&E staining, hematoxylin, nucleus, morphology)
- 설명은 한국어로 작성
- 예시: "Hematoxylin은 세포핵을 보라색으로 염색합니다"
- 실제 사용자가 세포 이미지를 보고 궁금해할 만한 질문으로

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이:
[
  {{"instruction": "질문", "output": "답변"}},
  ...
]"""

os.makedirs("data", exist_ok=True)

results = []

for category in categories:
    print(f"생성 중: {category}")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt_template.format(category=category)}]
    )
    
    text = response.content[0].text
    try:
        items = json.loads(text)
        results.extend(items)
        print(f"  {len(items)}개 생성 완료")
    except Exception as e:
        print(f"  파싱 실패: {e}")

with open("data/sft_domain.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n총 {len(results)}개 저장 완료")