from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
import base64
from PIL import Image
import io
import os
import colorsys
import imageio
from inference import CellSAM

app = Flask(__name__)

# 모델 로드
model = CellSAM(
    sam_checkpoint='/workspace/sam_vit_b_01ec64.pth',
    cellfinder_checkpoint='/workspace/LeeJeongmin-project/cellsam/outputs/checkpoint_best.pth',
    neck_checkpoint='/workspace/LeeJeongmin-project/cellsam/outputs/neck_checkpoint_best.pth'
)

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
    writer = imageio.get_writer(output_path, fps=10)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)