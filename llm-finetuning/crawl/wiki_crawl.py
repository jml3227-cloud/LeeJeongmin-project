import wikipediaapi
import json
import os

wiki_en = wikipediaapi.Wikipedia(
    language='en',
    user_agent='LLM-finetuning-crawler/1.0 (jml3227@gmail.com)'
)

wiki_ko = wikipediaapi.Wikipedia(
    language='ko',
    user_agent='LLM-finetuning-crawler/1.0 (jml3227@gmail.com)'
)

queries_en = [
    "Cell (biology)",
    "Bacterial morphology",
    "Histopathology",
    "H&E stain",
    "Microscopy",
    "Pathology",
    "Gram stain",
    "Cell nucleus",
    "Cytology",
]

queries_ko = [
    "세포",
    "박테리아",
    "조직병리학",
    "헤마톡실린",
    "현미경",
    "병리학",
    "그람 염색",
    "세포핵",
    "세포생물학",
]

os.makedirs("data", exist_ok=True)

results = []

for query in queries_en:
    print(f"Fetching EN: {query}")
    page = wiki_en.page(query)
    if not page.exists():
        print(f"존재하지 않는 페이지: {query}")
        continue
    results.append({
        "title": page.title,
        "lang": "en",
        "text": page.text
    })
    print(f"완료: {len(page.text)}자")

for query in queries_ko:
    print(f"Fetching KO: {query}")
    page = wiki_ko.page(query)
    if not page.exists():
        print(f"존재하지 않는 페이지: {query}")
        continue
    results.append({
        "title": page.title,
        "lang": "ko",
        "text": page.text
    })
    print(f"완료: {len(page.text)}자")

with open("data/wiki_raw.jsonl", "w") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n총 {len(results)}개 저장 완료")