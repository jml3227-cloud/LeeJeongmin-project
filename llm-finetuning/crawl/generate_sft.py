import json
import os
from google import genai

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

categories = [
    "H&E staining 색깔과 의미",
    "세포핵 형태학",
    "박테리아 형태 분류",
    "병리학적 소견",
    "현미경 이미지 해석",
    "Gram stain 원리와 해석",
    "세포막과 세포벽 구조",
    "조직 염색 방법론",
    "암세포 형태학적 특징",
    "세포 분열과 mitosis",
    "면역세포 형태 분류",
    "세포 괴사와 apoptosis",
    "조직 섬유화와 collagen",
    "혈구 형태학",
    "세포질 구조와 organelle",
    "병리 슬라이드 판독",
    "세포 염색 artifact",
    "핵분열 지수 평가",
    "조직 절편 두께와 염색 강도",
    "디지털 병리학 이미지 분석",
    "종양 세포의 핵 이형성",
    "세포 크기와 핵세포질 비율",
    "조직 침윤과 전이 소견",
    "유사분열 소견 해석",
    "세포막 투과성과 염색",
    "면역조직화학 염색 원리",
    "세포 집단과 군집 형태",
    "괴사 조직 병리 소견",
    "림프구 형태와 분류",
    "상피세포 형태학",
    "결합 조직 세포 형태",
    "세균 세포벽 구조와 항생제",
    "구균과 간균의 형태 비교",
    "진균 형태학",
    "바이러스 감염 세포 변화",
    "혈액 도말 검사 판독",
    "골수 세포 형태학",
    "종양 분화도 평가",
    "세포 부착과 세포외기질",
    "염증 세포 침윤 소견",
    "선암과 편평세포암 비교",
    "세포 주기와 암 발생",
    "종양 미세환경 세포 구성",
    "세포 노화와 형태 변화",
    "줄기세포 형태학적 특징",
    "세포 이주와 침습 기전",
    "병리 검체 처리 방법론",
    "냉동 절편과 파라핀 절편 비교",
    "전자현미경으로 보는 세포 구조",
    "세포 사멸 경로와 형태적 변화",
]
 
prompt_template = """당신은 세포 형태학과 병리학 전문가입니다.
아래 주제에 대한 질문-답변 쌍 10개를 생성해주세요.
 
주제: {category}
 
규칙:
- 전문 용어는 영어로 사용 (예: H&E staining, hematoxylin, nucleus, morphology)
- 설명은 한국어로 작성
- 실제 사용자가 세포 이미지를 보고 궁금해할 만한 질문으로
 
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이:
[
  {{"instruction": "질문", "output": "답변"}},
  ...
]"""
 
os.makedirs("data", exist_ok=True)
 
total = 0
 
with open("data/sft_domain.jsonl", "a", encoding="utf-8") as f:
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