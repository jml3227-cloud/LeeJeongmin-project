import torch
import numpy as np
import av
import argparse
import os

from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
from peft import PeftModel


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def load_video(video_path, num_frames=4):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        stream = container.streams.video[0]
        total_frames = int(stream.duration * stream.average_rate)
    indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)
    clip = read_video_pyav(container, indices)
    return clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="base model path")
    parser.add_argument("--adapter_path", type=str, required=True, help="LoRA adapter path")
    parser.add_argument("--video_path", type=str, required=True, help="mp4 클립 경로")
    parser.add_argument("--num_frames", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    projector_path = os.path.join(args.adapter_path, "projector_fixed2.bin")
    if os.path.exists(projector_path):
        projector_state = torch.load(projector_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(projector_state, strict=False)
        print(f"Projector loaded. missing: {len(missing)}, unexpected: {len(unexpected)}")

    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    print(f"Loading video: {args.video_path}")
    clip = load_video(args.video_path, args.num_frames)  # (num_frames, H, W, 3)

    # conversation 구성
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "현재 수행 중인 실험 task와 전체 실험 대비 완료율을 알려주세요."}
            ]
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        videos=[clip],
        return_tensors="pt"
    ).to(device, torch.bfloat16)

    print("Generating...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    generated = processor.decode(output[0], skip_special_tokens=True)
    # assistant 응답만 추출
    if "assistant" in generated:
        generated = generated.split("assistant")[-1].strip()

    print(f"\n=== 결과 ===")
    print(f"비디오: {args.video_path}")
    print(f"모델 출력: {generated}")


if __name__ == "__main__":
    main()