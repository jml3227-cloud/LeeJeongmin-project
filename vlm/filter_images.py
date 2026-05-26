import json
import os
import shutil

SAMPLED_JSON = os.path.expanduser("/home/userjml3227/sampled_dataset.json")
# 이미지가 저장된 원본 폴더들 (로컬 기준)
SRC_DIRS = [
    "/home/userjml3227/vlm_raw/10.위암_병리_이미지_및_판독문_합성데이터/3.개방데이터/1.데이터/Training/01.원천데이터",
    "/home/userjml3227/vlm_raw/11.유방암_병리_이미지_및_판독문_합성데이터/3.개방데이터/1.데이터/Training/01.원천데이터",
]
DST_DIR = "/home/userjml3227/vlm_images"


def main():
    with open(SAMPLED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 필요한 file_name 추출
    file_names = set()
    for category, entries in data.items():
        for entry in entries:
            file_names.add(entry["file_name"])

    print(f"필요한 이미지 수: {len(file_names)}")

    os.makedirs(DST_DIR, exist_ok=True)

    count = 0
    not_found = []

    for src_dir in SRC_DIRS:
        if not os.path.exists(src_dir):
            print(f"폴더 없음: {src_dir}")
            continue
        for fname in os.listdir(src_dir):
            if fname in file_names:
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(DST_DIR, fname)
                shutil.copy2(src_path, dst_path)
                count += 1

    # 못 찾은 파일 확인
    found = set(os.listdir(DST_DIR))
    not_found = file_names - found
    if not_found:
        print(f"\n못 찾은 파일 {len(not_found)}개:")
        for f in list(not_found)[:10]:
            print(f"  {f}")

    print(f"\n복사 완료: {count}개")
    print(f"저장 위치: {DST_DIR}")


if __name__ == "__main__":
    main()