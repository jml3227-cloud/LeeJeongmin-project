import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import resize
from cellsam_models.anchor_detr import build
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/workspace/LeeJeongmin-project/cellsam/outputs/checkpoint_epoch9.pth', type=str)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_path', default='result.png', type=str)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--device', default='cuda', type=str)

    # build에 필요한 args
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--num_feature_levels', default=1, type=int)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--masks', default=False, type=bool)
    parser.add_argument('--sam_checkpoint', default='/workspace/LeeJeongmin-project/cellsam/sam_vit_b_01ec64.pth', type=str)
    
    parser.add_argument('--only_neck', default=False, type=bool)
    parser.add_argument('--freeze_backbone', default=False, type=bool)
    parser.add_argument('--cls_loss_coef', default=2.0, type=float)
    parser.add_argument('--bbox_loss_coef', default=5.0, type=float)
    parser.add_argument('--giou_loss_coef', default=2.0, type=float)
    parser.add_argument('--mask_loss_coef', default=1.0, type=float)
    parser.add_argument('--dice_loss_coef', default=1.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--dec_layers', default=6, type=int)

    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_query_position', default=3500, type=int)
    parser.add_argument('--num_query_pattern', default=1, type=int)
    parser.add_argument('--spatial_prior', default='learned', type=str)
    parser.add_argument('--attention_type', default='RCDA', type=str)

    parser.add_argument('--set_cost_class', default=2.0, type=float)
    parser.add_argument('--set_cost_bbox', default=5.0, type=float)
    parser.add_argument('--set_cost_giou', default=2.0, type=float)
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 불러오기
    model, _, postprocessors = build(args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # 이미지 불러오기
    image = np.array(Image.open(args.image_path).convert('RGB'))
    H, W = image.shape[:2]
    image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    image_tensor = resize(image_tensor, [1024, 1024]).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        outputs = model(image_tensor)

    # bounding box 후처리
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]

    scores = pred_logits.sigmoid().max(-1).values
    print(f"최대 score: {scores.max().item():.4f}")
    print(f"평균 score: {scores.mean().item():.4f}")
    print(f"score 표준편차: {scores.std().item():.4f}")
    
    keep = scores > args.threshold

    print(f"keep 개수: {keep.sum().item()}")

    boxes = pred_boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()

    # cx, cy, w, h -> x_min, y_min, x_max, y_max 변환
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H

    # 시각화
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(image)
    for box in boxes_xyxy:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(f'검출된 세포 수: {len(boxes)}')
    ax.axis('off')
    plt.savefig(args.output_path)
    plt.close()
    print(f"저장완료: {args.output_path}, 검출 세포 수: {len(boxes)}")


if __name__ == "__main__":
    main()