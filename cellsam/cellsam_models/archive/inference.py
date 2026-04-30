import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import resize
from cellsam_models.AnchorDETR.models.anchor_detr import build
from cellsam_models.train.dataset import MoNuSACDataset, TNBCDataset, NuInsSegDataset, collate_fn
from torch.utils.data import DataLoader, ConcatDataset
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='/workspace/LeeJeongmin-project/cellsam/outputs/checkpoint_best.pth', type=str)
    parser.add_argument('--mode', default='visualize', choices=['visualize', 'eval'])
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_path', default='result.png', type=str)
    parser.add_argument('--threshold', default=0.4, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--data_dir', default='/workspace', type=str)

    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--num_feature_levels', default=1, type=int)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--masks', default=False, type=bool)
    parser.add_argument('--sam_checkpoint', default='/workspace/sam_vit_b_01ec64.pth', type=str)
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

def box_iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

def compute_f1(pred_boxes, gt_boxes, iou_threshold=0.5):
    tp = 0
    matched = set()
    for pred in pred_boxes:
        for j, gt in enumerate(gt_boxes):
            if j in matched:
                continue
            if box_iou(pred, gt) >= iou_threshold:
                tp += 1
                matched.add(j)
                break
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def predict(model, image_path, args, device):
    image = np.array(Image.open(image_path).convert('RGB'))
    H, W = image.shape[:2]
    image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
    image_tensor = resize(image_tensor, [1024, 1024]).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]
    scores = pred_logits.sigmoid().max(-1).values
    keep = scores > args.threshold
    boxes = pred_boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
    return image, boxes_xyxy, scores

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, _, postprocessors = build(args)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    if args.mode == 'visualize':
        assert args.image_path is not None
        image, boxes_xyxy, scores = predict(model, args.image_path, args, device)
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        for box in boxes_xyxy:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                      linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.set_title(f'검출 세포 수: {len(boxes_xyxy)}')
        ax.axis('off')
        plt.savefig(args.output_path)
        plt.close()
        print(f'저장완료: {args.output_path}, 검출 세포 수: {len(boxes_xyxy)}')

    elif args.mode == 'eval':
        monusac_test = MoNuSACDataset(os.path.join(args.data_dir, 'monusac'), split='test')
        tnbc_test = TNBCDataset(os.path.join(args.data_dir, 'tnbc'), split='test')
        nuinsseg_test = NuInsSegDataset(os.path.join(args.data_dir, 'nuinsseg'), split='test')
        test_dataset = ConcatDataset([monusac_test, tnbc_test, nuinsseg_test])
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        all_precision, all_recall, all_f1 = [], [], []

        for images, targets in test_dataloader:
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
            pred_logits = outputs['pred_logits'][0]
            pred_boxes_raw = outputs['pred_boxes'][0]
            scores = pred_logits.sigmoid().max(-1).values
            keep = scores > args.threshold
            pred_boxes = pred_boxes_raw[keep].cpu().numpy()
            gt_boxes = targets[0]['boxes'].cpu().numpy()

            pred_xyxy = np.zeros_like(pred_boxes)
            pred_xyxy[:, 0] = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
            pred_xyxy[:, 1] = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
            pred_xyxy[:, 2] = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
            pred_xyxy[:, 3] = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

            gt_xyxy = np.zeros_like(gt_boxes)
            gt_xyxy[:, 0] = gt_boxes[:, 0] - gt_boxes[:, 2] / 2
            gt_xyxy[:, 1] = gt_boxes[:, 1] - gt_boxes[:, 3] / 2
            gt_xyxy[:, 2] = gt_boxes[:, 0] + gt_boxes[:, 2] / 2
            gt_xyxy[:, 3] = gt_boxes[:, 1] + gt_boxes[:, 3] / 2

            precision, recall, f1 = compute_f1(pred_xyxy.tolist(), gt_xyxy.tolist())
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)

        print(f'Precision: {np.mean(all_precision):.4f}')
        print(f'Recall: {np.mean(all_recall):.4f}')
        print(f'F1 Score: {np.mean(all_f1):.4f}')

if __name__ == "__main__":
    main()
