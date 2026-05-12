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

    masks = model.predict(img_tensor)
    mask = masks[0]

    # 세포 개수
    cell_count = int(mask.max())

    # 마스크 시각화
    mask_b64 = visualize_mask(img, mask)

    return jsonify({
        'mask_image': mask_b64,
        'cell_count': cell_count
    })

def visualize_mask(img, mask):
    overlay = img.copy()
    np.random.seed(42)
    colors = np.random.randint(0, 255, (mask.max() + 1, 3))
    for cell_id in range(1, mask.max() + 1):
        overlay[mask == cell_id] = colors[cell_id]
    result = (img * 0.4 + overlay * 0.6).astype(np.uint8)

    pil_image = Image.fromarray(result)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    mask_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return mask_b64

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)