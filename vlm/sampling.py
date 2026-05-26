import json
import os
import random

def load_json(filepath):
    with open(filepath, "rb") as f:
        return json.loads(f.read().decode("utf-8-sig"))

def sample_dataset(breast_dir, gastric_dir, output_path, n=1000, seed=42):
    random.seed(seed)

    gastric_inflammation = []
    gastric_cancer = []
    breast_normal = []
    breast_cancer = []

    # 위암 데이터
    for fname in os.listdir(gastric_dir):
        if not fname.endswith(".json"):
            continue
        d = load_json(os.path.join(gastric_dir, fname))
        c = d["content"]
        category = c["clinical"]["category"]
        desc = c["file"]["patch_discription"]
        entry = {
            "file_name": c["file"]["file_name"],
            "patch_discription": desc,
            "tumor_category": c["clinical"]["tumor_category"]
        }
        if category == "STNT":
            if "inflammation" in desc.lower():
                gastric_inflammation.append(entry)
        else:
            gastric_cancer.append(entry)

    # 유방암 데이터
    for fname in os.listdir(breast_dir):
        if not fname.endswith(".json"):
            continue
        d = load_json(os.path.join(breast_dir, fname))
        c = d["content"]
        category = c["clinical"]["category"]
        desc = c["file"]["patch_discription"]
        entry = {
            "file_name": c["file"]["file_name"],
            "patch_discription": desc,
            "tumor_category": c["clinical"]["tumor_category"]
        }
        if category == "BRNT":
            breast_normal.append(entry)
        else:
            breast_cancer.append(entry)

    # 샘플링
    sampled = {
        "gastric_inflammation": random.sample(gastric_inflammation, min(n, len(gastric_inflammation))),
        "gastric_cancer": random.sample(gastric_cancer, min(n, len(gastric_cancer))),
        "breast_normal": random.sample(breast_normal, min(n, len(breast_normal))),
        "breast_cancer": random.sample(breast_cancer, min(n, len(breast_cancer)))
    }

    for k, v in sampled.items():
        print(f"{k}: {len(v)}개")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료: {output_path}")

if __name__ == "__main__":
    sample_dataset(
        breast_dir=os.path.expanduser("~/breast_cancer_json"),
        gastric_dir=os.path.expanduser("~/gastric_cancer_json"),
        output_path=os.path.expanduser("~/sampled_dataset.json"),
        n=1000
    )