import json
import subprocess
import sys
import re
from pathlib import Path


def parse_output(generated):
    """모델 출력에서 task명과 완료율 추출"""
    task = None
    completion = None
    
    # task 추출
    task_match = re.search(r'현재 task[:\s]+([^\n]+)', generated)
    if task_match:
        task = task_match.group(1).strip()
    
    # 완료율 추출
    completion_match = re.search(r'완료율[:\s]+([\d.]+)', generated)
    if completion_match:
        completion = float(completion_match.group(1))
    
    return task, completion


def main():
    valid_json = '/workspace/LeeJeongmin-project/vlm/data/finebio_valid.json'
    video_folder = '/workspace/LeeJeongmin-project/vlm/data/vlm_videos'
    model_path = '/workspace/llava-onevision-qwen2-7b-ov-hf'
    adapter_path = '/workspace/LeeJeongmin-project/vlm/outputs/finebio'

    with open(valid_json) as f:
        data = json.load(f)

    total = len(data)
    task_correct = 0
    completion_errors = []
    failed = 0

    print(f"총 val 샘플 수: {total}")
    print("평가 시작...\n")

    for i, sample in enumerate(data):
        video_path = f"{video_folder}/{sample['video']}"
        gt_text = sample['conversations'][1]['value']

        # 정답 파싱
        gt_task_match = re.search(r'현재 task[:\s]+([^\n]+)', gt_text)
        gt_completion_match = re.search(r'완료율[:\s]+([\d.]+)', gt_text)

        if not gt_task_match or not gt_completion_match:
            failed += 1
            continue

        gt_task = gt_task_match.group(1).strip()
        gt_completion = float(gt_completion_match.group(1))

        # 추론
        result = subprocess.run(
            [
                sys.executable,
                '/workspace/LeeJeongmin-project/vlm/finebio_inference.py',
                '--model_path', model_path,
                '--adapter_path', adapter_path,
                '--video_path', video_path,
                '--num_frames', '8'
            ],
            capture_output=True,
            text=True
        )

        output = result.stdout
        generated = ""
        if "모델 출력:" in output:
            generated = output.split("모델 출력:")[-1].strip()

        pred_task, pred_completion = parse_output(generated)

        # task 정확도
        if pred_task and pred_task.strip() == gt_task:
            task_correct += 1

        # 완료율 MAE
        if pred_completion is not None:
            completion_errors.append(abs(pred_completion - gt_completion))

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{total}] task 정확도: {task_correct/(i+1)*100:.1f}%, MAE: {sum(completion_errors)/len(completion_errors):.1f}%")

    # 최종 결과
    task_accuracy = task_correct / total * 100
    mae = sum(completion_errors) / len(completion_errors) if completion_errors else 0

    print(f"\n========== 최종 결과 ==========")
    print(f"총 샘플: {total}")
    print(f"task 정확도: {task_correct}/{total} ({task_accuracy:.1f}%)")
    print(f"완료율 MAE: {mae:.1f}%")
    print(f"파싱 실패: {failed}")


if __name__ == "__main__":
    main()