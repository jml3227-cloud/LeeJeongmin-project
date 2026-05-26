import json
import os
import sys
import numpy as np
import torch
from PIL import Image

# ----------------------------------------
# 경로 설정 (RunPod 기준)
# ----------------------------------------
IMAGE_DIR = "/workspace/LeeJeongmin-project/vlm/data/vlm_images"
OUTPUT_JSON = "/workspace/LeeJeongmin-project/vlm/data/cellsam_results.json"

SAM_CHECKPOINT = "/workspace/sam_vit_b_01ec64.pth"
CELLFINDER_CHECKPOINT = "/workspace/LeeJeongmin-project/cellsam/outputs/checkpoint_best.pth"
NECK_CHECKPOINT = "/workspace/LeeJeongmin-project/cellsam/outputs/neck_checkpoint_best.pth"

sys.path.insert(0, "/workspace/LeeJeongmin-project/cellsam")


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    return img_tensor


def extract_quantification(mask):
    """instance mask에서 세포 수, 밀도, 면적 표준편차 추출"""
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids > 0]  # 배경(0) 제외

    cell_count = len(unique_ids)
    total_pixels = mask.shape[0] * mask.shape[1]

    if cell_count == 0:
        return {
            "cell_count": 0,
            "density": 0.0,
            "std_area": 0.0
        }

    areas = [np.sum(mask == uid) for uid in unique_ids]
    density = cell_count / total_pixels
    std_area = float(np.std(areas))

    return {
        "cell_count": cell_count,
        "density": round(density, 6),
        "std_area": round(std_area, 2)
    }


def main():
    from cellsam_models.cellsam_inference import CellSAM

    model = CellSAM(
        sam_checkpoint=SAM_CHECKPOINT,
        cellfinder_checkpoint=CELLFINDER_CHECKPOINT,
        neck_checkpoint=NECK_CHECKPOINT,
        device="cuda"
    )
    model.eval()

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(".png")]
    total = len(image_files)
    print(f"총 이미지 수: {total}")

    results = {}
    failed = []

    for i, fname in enumerate(image_files):
        image_path = os.path.join(IMAGE_DIR, fname)
        print(f"[{i+1}/{total}] {fname}")

        try:
            img_tensor = load_image(image_path)
            masks = model.predict(img_tensor)
            quant = extract_quantification(masks[0])
            results[fname] = quant
        except Exception as e:
            print(f"  실패: {e}")
            failed.append(fname)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n완료: {len(results)}개 저장 → {OUTPUT_JSON}")
    if failed:
        print(f"실패: {len(failed)}개")
        for f in failed[:10]:
            print(f"  {f}")


if __name__ == "__main__":
    main()