import os
import json
import csv
import subprocess
from pathlib import Path

# ==================== 경로 설정 ====================
VIDEO_DIR = os.path.expanduser("~/finebio_videos_fpv_all_w640/finebio_videos_w640")
ANNOT_DIR = os.path.expanduser("~/annotations/annotations/finebio_action_annotations/finebio_action_annotations")
CLIP_DIR = os.path.expanduser("~/finebio_clips")
JSON_PATH = os.path.expanduser("~/finebio_dataset.json")
# ===================================================

os.makedirs(CLIP_DIR, exist_ok=True)


def parse_annotation(txt_path):
    """annotation txt에서 task 레벨 행만 추출"""
    tasks = []
    with open(txt_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # task 컬럼에 값 있으면 task 레벨 행
            if row["task"].strip():
                tasks.append({
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "task": row["task"].strip()
                })
    return tasks


def cut_clip(video_path, start_sec, end_sec, out_path):
    """ffmpeg으로 구간 잘라서 저장"""
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-an",          # 오디오 제거
        "-loglevel", "error",
        out_path
    ]
    subprocess.run(cmd, check=True)


def main():
    dataset = []
    video_files = sorted(Path(VIDEO_DIR).glob("*.mp4"))
    total = len(video_files)

    print(f"총 비디오 수: {total}")

    for idx, video_path in enumerate(video_files):
        stem = video_path.stem  # ex) P01_01_01
        annot_path = os.path.join(ANNOT_DIR, f"{stem}.txt")

        if not os.path.exists(annot_path):
            print(f"[SKIP] annotation 없음: {stem}")
            continue

        tasks = parse_annotation(annot_path)
        if not tasks:
            print(f"[SKIP] task 없음: {stem}")
            continue

        # total_duration = 마지막 task의 end_sec
        total_duration = tasks[-1]["end_sec"]

        print(f"[{idx+1}/{total}] {stem} - task {len(tasks)}개")

        for task_idx, task in enumerate(tasks):
            clip_name = f"{stem}_task{task_idx:02d}_{task['task']}.mp4"
            clip_path = os.path.join(CLIP_DIR, clip_name)

            # 클립 잘라서 저장
            if not os.path.exists(clip_path):
                try:
                    cut_clip(str(video_path), task["start_sec"], task["end_sec"], clip_path)
                except subprocess.CalledProcessError as e:
                    print(f"  [ERROR] 클립 저장 실패: {clip_name}")
                    continue

            # 완료율 계산
            completion = round(task["end_sec"] / total_duration * 100, 1)

            # JSON 샘플 생성
            sample = {
                "video": clip_name,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>\n현재 수행 중인 실험 task와 전체 실험 대비 완료율을 알려주세요."
                    },
                    {
                        "from": "gpt",
                        "value": f"현재 task: {task['task']}\n완료율: {completion}%"
                    }
                ]
            }
            dataset.append(sample)

    # JSON 저장
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n완료! 총 샘플 수: {len(dataset)}")
    print(f"클립 저장 위치: {CLIP_DIR}")
    print(f"JSON 저장 위치: {JSON_PATH}")


if __name__ == "__main__":
    main()