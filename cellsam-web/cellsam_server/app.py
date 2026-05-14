from flask import Flask, request, jsonify
import torch
import numpy as np
import base64
from PIL import Image
import io
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

    masks, avg_scores = model.predict(img_tensor)
    avg_confidence = round(avg_scores[0], 4)
    mask = masks[0]

    # 세포 개수
    cell_count = int(mask.max())

    # 마스크 시각화
    mask_b64 = visualize_mask(img, mask)

    return jsonify({
        'mask_image': mask_b64,
        'cell_count': cell_count,
        'avg_confidence': avg_confidence
    })

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
        result[mask == cell_id] = colors[cell_id]

    pil_image = Image.fromarray(result)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)