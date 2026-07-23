from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
import base64
from PIL import Image
import io
import os
import colorsys
import imageio
import re
import av
from inference import CellSAM
from transformers import AutoTokenizer, AutoModelForCausalLM, LlavaOnevisionForConditionalGeneration, AutoProcessor
from peft import PeftModel

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# 모델 로드
model = CellSAM(
    sam_checkpoint = '/workspace/sam_vit_b_01ec64.pth',
    cellfinder_checkpoint = '/workspace/LeeJeongmin-project/cellsam/outputs_full_norm/checkpoint_best.pth',
    neck_checkpoint = '/workspace/LeeJeongmin-project/cellsam/outputs_full_norm/neck_checkpoint_best.pth'
)
model.bbox_threshold = 0.4
model.iou_threshold = 0.4

LLM_MODEL_PATH = '/workspace/LeeJeongmin-project/llm-finetuning/outputs/qlora_final_v2'
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
llm_tokenizer.pad_token = llm_tokenizer.eos_token
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

VLM_BASE_PATH = '/workspace/llava-onevision-qwen2-7b-ov-hf'
VLM_ADAPTER_PATH = '/workspace/LeeJeongmin-project/vlm/outputs/checkpoints_rebuilt'
FINEBIO_ADAPTER_PATH = '/workspace/LeeJeongmin-project/finebio/outputs/checkpoints'

processor = AutoProcessor.from_pretrained(VLM_BASE_PATH)
base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    VLM_BASE_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

vlm_model = PeftModel.from_pretrained(base_model, VLM_ADAPTER_PATH, adapter_name="vlm")
vlm_model.load_adapter(FINEBIO_ADAPTER_PATH, adapter_name="finebio")
vlm_model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다'}), 400
    
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = np.array(img)

    # (H, W, 3) -> (3, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    masks, avg_ious = model.predict(img_tensor)
    avg_iou = avg_ious[0]
    mask = masks[0]

    # 세포 개수
    cell_count = int(mask.max())

    # 마스크 시각화
    frame = visualize_mask(img, mask)
    mask_b64 = numpy_to_b64(frame)

    return jsonify({
        'mask_image': mask_b64,
        'cell_count': cell_count,
        'avg_iou': avg_iou
    })

@app.route('/predict_video', methods=['POST'])
def predict_video():
    if 'images' not in request.files:
        return jsonify({'error': '이미지가 없습니다'}), 400
    
    files = request.files.getlist('images')
    files = sorted(files, key=lambda f: f.filename)

    frames = []
    prev_masks = None
    prev_ids = None

    for file in files:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = np.array(img)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        masks, _ = model.predict(img_tensor)
        mask = masks[0]

        # IoU 기반 세포 ID 매칭
        mask, prev_masks, prev_ids = match_ids(mask, prev_masks, prev_ids)

        frame = visualize_mask(img, mask)
        
        frames.append(frame)

    # MP4 만들기
    output_path = '/tmp/result.mp4'
    writer = imageio.get_writer(output_path, format='mp4', fps=4)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    return send_file(output_path, mimetype='video/mp4')

def match_ids(mask, prev_masks, prev_ids):
    if prev_masks is None:
        n = mask.max()
        ids = list(range(1, n+1))
        return mask, mask.copy(), ids
    
    new_mask = np.zeros_like(mask)
    used_ids = set()
    next_id = max(prev_ids) + 1 if prev_ids else 1

    for new_label in range(1, mask.max() + 1):
        new_region = (mask == new_label)
        best_iou = 0
        best_prev_id = None

        for prev_id in prev_ids:
            prev_region = (prev_masks == prev_id)
            intersection = np.logical_and(new_region, prev_region).sum()
            union = np.logical_or(new_region, prev_region).sum()
            if union == 0:
                continue
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_prev_id = prev_id

        if best_iou > 0.3 and best_prev_id is not None and best_prev_id not in used_ids:
            new_mask[new_region] = best_prev_id
            used_ids.add(best_prev_id)
        else:
            new_mask[new_region] = next_id
            next_id += 1

    new_ids = [int(i) for i in np.unique(new_mask) if i != 0]
    return new_mask, new_mask.copy(), new_ids

def visualize_mask(img, mask):
    import colorsys
    result = np.zeros_like(img)
    n_cells = mask.max()
    
    colors = []
    for i in range(n_cells + 1):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((int(r*255), int(g*255), int(b*255)))
    colors = np.array(colors)

    for cell_id in range(1, n_cells + 1):
        if cell_id < len(colors):
            result[mask == cell_id] = colors[cell_id]

    return result

def numpy_to_b64(arr):
    pil_image = Image.fromarray(arr)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

@app.route('/llm/generate', methods=['POST'])
def llm_generate():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': '질문이 없습니다'})
    
    question = data['question']
    history = data.get('history', [])

    history_text = ''
    for turn in history:
        if turn['role'] == 'user':
            history_text += f"### 질문:\n{turn['content']}\n\n"
        elif turn['role'] == 'assistant':
            history_text += f"### 답변:\n{turn['content']}\n\n"

    prompt = f"간결하게 1~2문장으로 답하세요.\n\n{history_text}### 질문:\n{question}\n\n### 답변:"

    inputs = llm_tokenizer(prompt, return_tensors='pt').to('cuda')
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=300,
        repetition_penalty=1.5,
        do_sample=True,
        temperature=0.1,
        eos_token_id=llm_tokenizer.eos_token_id,
    )

    full_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = full_text.split('### 답변:')[-1].strip()
    sentences = re.split(r'(?<=[다요])\s', answer)
    answer = ' '.join(sentences[:2]).strip()
    answer = re.split(r'\n[A-Za-z]', answer)[0].strip()
    answer = re.split(r'\s{2,}[A-Z][a-z]', answer)[0].strip()
    answer = re.sub(r'\(https?://\S+\)', '', answer).strip()
    answer = re.sub(r'https?://\S+', '', answer).strip()

    return jsonify({'answer': answer})

@app.route('/vlm/analyze', methods=['POST'])
def vlm_analyze():
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다'}), 400
    
    file = request.files['image']
    question = request.form.get('question', '이 조직 슬라이드 소견을 말해주세요.')

    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_np = np.array(img)

    # cellsam으로 정량 데이터 추출
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    masks, _ = model.predict(img_tensor)
    mask = masks[0]

    cell_count = int(mask.max())
    total_pixels = mask.shape[0] * mask.shape[1]
    cell_pixels = int((mask > 0).sum())
    density = round(cell_pixels / total_pixels, 4)

    cell_areas = []
    for cell_id in range(1, cell_count + 1):
        area = int((mask == cell_id).sum())
        if area > 0:
            cell_areas.append(area)
    std_area = round(float(np.std(cell_areas)), 1) if cell_areas else 0.0

    user_text = (
        f"[세포 분석 결과] 세포 수: {cell_count}개, "
        f"밀도: {density:.4f}/px²\n"
        f"{question}"
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": user_text}
        ]
    }]

    vlm_model.set_adapter("vlm")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=img, text=prompt, return_tensors="pt").to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = vlm_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    generated = processor.decode(outputs[0], skip_special_tokens=True)
    answer = generated.split("assistant\n")[-1].strip()

    mask_frame = visualize_mask(img_np, mask)
    mask_b64 = numpy_to_b64(mask_frame)

    return jsonify({
        'answer': answer,
        'cell_count': cell_count,
        'density': density,
        'std_area': std_area,
        'mask_image': mask_b64
    })

@app.route('/vlm/chat', methods=['POST'])
def vlm_chat():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': '질문이 없습니다'}), 400
    
    question = data['question']
    history = data.get('history', [])

    messages = []
    for turn in history:
        role = turn['role']
        messages.append({
            "role": role,
            "content": [{"type": "text", "text": turn['content']}]
        })

    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": question}]
    })

    vlm_model.set_adapter("vlm")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = vlm_model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated = processor.decode(outputs[0], skip_special_tokens=True)
    answer = generated.split("assistant\n")[-1].strip()

    return jsonify({'answer': answer})

@app.route('/finebio/analyze', methods=['POST'])
def finebio_analyze():
    if 'video' not in request.files:
        return jsonify({'error': '비디오가 없습니다'}), 400
    
    file = request.files['video']

    tmp_path = '/tmp/finebio_input.mp4'
    file.save(tmp_path)

    frames = []
    container = av.open(tmp_path)
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    indices = set(np.linspace(0, total_frames - 1, 8, dtype=int))

    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            img = frame.to_image().convert('RGB')
            frames.append(img)
        if len(frames) == 8:
            break
    container.close()

    if not frames:
        return jsonify({'error': '프레임 추출 실패'}), 400
    
    content = []
    for _ in frames:
        content.append({"type": "image"})
    content.append({"type": "text", "text": "현재 수행 중인 실험 task와 전체 실험 대비 완료율을 알려주세요."})

    messages = [{"role": "user", "content": content}]
    vlm_model.set_adapter("finebio")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=frames, text=prompt, return_tensors="pt").to("cuda")
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    with torch.no_grad():
        outputs = vlm_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    generated = processor.decode(outputs[0], skip_special_tokens=True)
    answer = generated.split("assistant\n")[-1].strip()

    task_name = ''
    completion_rate = 0.0

    for line in answer.split('\n'):
        if '현재 task' in line:
            task_name = line.split('현재 task:')[-1].strip()
        if '완료율' in line:
            rate_str = line.split('완료율:')[-1].strip().replace('%', '').strip()
            try:
                completion_rate = float(rate_str)
            except ValueError:
                completion_rate = 0.0

    return jsonify({
        'task_name': task_name,
        'completion_rate': completion_rate
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)