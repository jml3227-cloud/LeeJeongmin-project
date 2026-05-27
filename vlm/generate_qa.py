import json
import os
import time
from openai import OpenAI

# ----------------------------------------
# 경로 설정 (RunPod 기준)
# ----------------------------------------
SAMPLED_JSON = "/workspace/LeeJeongmin-project/vlm/data/sampled_dataset.json"
CELLSAM_JSON = "/workspace/LeeJeongmin-project/vlm/data/cellsam_results.json"
IMAGE_DIR = "/workspace/LeeJeongmin-project/vlm/data/images"
OUTPUT_JSON = "/workspace/LeeJeongmin-project/vlm/data/qa_dataset.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = (
    "당신은 H&E 염색 조직 이미지를 분석하는 병리 AI 어시스턴트입니다. "
    "아래 입력을 바탕으로 임상의와 병리 AI 간의 한국어 멀티턴 대화를 생성하세요.\n\n"
    "규칙:\n"
    "1. 첫 번째 human 턴은 조직 이미지 분석을 요청하는 자연스러운 한국어 질문으로 시작하세요. "
    "표현은 매번 다양하게 생성하세요. (예: '이 조직 슬라이드 소견을 말해주세요.', "
    "'조직 이미지 분석 부탁드립니다.', '이 검체의 조직학적 소견은 어떻습니까?')\n"
    "2. 첫 번째 gpt 턴은 주어진 판독문을 근거로 조직 소견을 한국어로 설명하세요. "
    "병리 용어는 영어 그대로 사용해도 됩니다. 세포 수, 밀도, 크기 정보도 함께 언급하세요.\n"
    "3. 이후 2-3턴은 임상의가 첫 번째 답변을 보고 추가로 물어볼 법한 질문과 답변으로 구성하세요.\n"
    "4. 세포 형태, 밀도, 크기, 조직 구조 등 조직학적 소견 중심으로 대화를 구성하세요.\n"
    "5. 마지막 gpt 턴에서는 조직학적 소견을 종합하여 이상 소견 여부를 간략히 요약하세요.\n"
    "6. JSON 형식으로만 응답하세요. 키는 'from'(human/gpt)과 'value'입니다. "
    "마크다운 백틱이나 추가 설명 없이 JSON만 출력하세요."
)


def make_user_prompt(description, cellsam_info):
    cell_count = cellsam_info.get("cell_count", "N/A")
    density = cellsam_info.get("density", "N/A")
    std_area = cellsam_info.get("std_area", "N/A")

    if isinstance(density, float):
        density = f"{density:.4f}"
    if isinstance(std_area, float):
        std_area = f"{std_area:.1f}"

    return (
        f"판독문: {description}\n"
        f"CellSAM 정량 결과: 세포 수={cell_count}개, "
        f"세포 밀도={density}/px², "
        f"세포 면적 표준편차={std_area}px²\n\n"
        f"위 정보를 바탕으로 멀티턴 대화를 생성하세요."
    )


def generate_qa(description, cellsam_info, retries=3):
    prompt = make_user_prompt(description, cellsam_info)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.7
            )
            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()
            conversations = json.loads(text)

            # 첫 번째 human 턴에 <image> + CellSAM 수치 추가
            if conversations and conversations[0]["from"] == "human":
                cell_count = cellsam_info.get("cell_count", "N/A")
                density = cellsam_info.get("density", "N/A")
                if isinstance(density, float):
                    density = f"{density:.4f}"
                conversations[0]["value"] = (
                    f"<image>\n"
                    f"[세포 분석 결과] 세포 수: {cell_count}개, "
                    f"밀도: {density}/px²\n"
                    + conversations[0]["value"]
                )

            return conversations

        except Exception as e:
            print(f"  재시도 {attempt+1}/{retries}: {e}")
            time.sleep(2)
    return None


def main():
    with open(SAMPLED_JSON, "r", encoding="utf-8") as f:
        sampled = json.load(f)

    with open(CELLSAM_JSON, "r", encoding="utf-8") as f:
        cellsam = json.load(f)  # {file_name: {cell_count, density, std_area}}

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    results = []
    total = sum(len(v) for v in sampled.values())
    count = 0

    for category, entries in sampled.items():
        for entry in entries[:3]:
            count += 1
            fname = entry["file_name"]
            desc = entry["patch_discription"]
            cellsam_info = cellsam.get(fname, {})

            print(f"[{count}/{total}] {fname}")

            conversations = generate_qa(desc, cellsam_info)
            if conversations is None:
                print(f"  실패: {fname}")
                continue

            results.append({
                "image": os.path.join(IMAGE_DIR, fname),
                "conversations": conversations
            })

            time.sleep(0.5)  # API rate limit 방지

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(results)}개 저장 → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()