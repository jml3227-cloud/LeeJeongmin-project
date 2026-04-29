import argparse
import os
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import torch
import matplotlib.pyplot as plt
from PIL import Image


from cellsam_models.cellsam_inference import CellSAM
from cellsam_models.utils import normalize_image, fill_holes_and_remove_small_masks
from cellsam_models.train.dataset import MoNuSACDataset, TNBCDataset, NuInsSegDataset, collate_fn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam_checkpoint', default='/workspace/sam_vit_b_01ec64.pth', type=str)
    parser.add_argument('--cellfinder_checkpoint', default='/workspace/LeeJeongmin-project/cellsam/outputs/checkpoint_best.pth', type=str)
    parser.add_argument('--neck_checkpoint', default='/workspace/LeeJeongmin-project/cellsam/outputs/neck_checkpoint_best.pth', type=str)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_path', default='/workspace/result_full.png', type=str)
    parser.add_argument('--mode', default='visualize', choices=['visualize', 'eval'])
    parser.add_argument('--data_dir', default='/workspace', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    return parser.parse_args()

def load_image(image_path):
    image = np.array(Image.open(image_path).convert('RGB'))
    return image

def visualize(image, mask, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(image)
    axes[0].set_title('원본 이미지')
    axes[0].axis('off')

    axes[1].imshow(image)
    axes[1].imshow(mask, alpha=0.5, cmap='jet')
    axes[1].set_title(f'검출 세포 수: {mask.max()}')
    axes[1].axis('off')

    plt.savefig(output_path)
    plt.close()
    print(f'저장완료: {output_path}')

def evaluate(model, data_dir, device):
    monusac_test = MoNuSACDataset(os.path.join(data_dir, 'monusac'), split='test')
    tnbc_test = TNBCDataset(os.path.join(data_dir, 'tnbc'), split='test')
    nuinsseg_test = NuInsSegDataset(os.path.join(data_dir, 'nuinsseg'), split='test')
    test_dataset = ConcatDataset([monusac_test, tnbc_test, nuinsseg_test])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    all_f1 = []
    for images, targets in test_loader:
        images = images.to(device)
        masks = model.predict(images)

        for idx in range(len(masks)):
            pred_mask = masks[idx]
            gt_mask = targets[idx]['masks'].cpu().numpy()

            # compute every instance
            pred_ids = np.unique(pred_mask)[1:]
            gt_ids = np.unique(gt_mask)[1:] if gt_mask.ndim == 2 else range(len(gt_mask))

            tp = 0
            matched = set()
            for pred_id in pred_ids:
                pred_bin = pred_mask == pred_id
                for j, gt_id in enumerate(gt_ids):
                    if j in matched:
                        continue
                    gt_bin = gt_mask[j] if gt_mask.ndim == 3 else gt_mask == gt_id
                    intersection = np.logical_and(pred_bin, gt_bin).sum()
                    union = np.logical_or(pred_bin, gt_bin).sum()
                    if union > 0 and intersection / union >= 0.5:
                        tp += 1
                        matched.add(j)
                        break
            
            fp = len(pred_ids) - tp
            fn = len(gt_ids) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            all_f1.append(f1)

    print(f'F1 Score: {np.mean(all_f1):.4f}')

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = CellSAM(
        sam_checkpoint=args.sam_checkpoint,
        cellfinder_checkpoint=args.cellfinder_checkpoint,
        neck_checkpoint=args.neck_checkpoint,
        device=str(device)
    )
    model.iou_threshold = args.iou_threshold
    model.eval()

    if args.mode == 'visualize':
        image = load_image(args.image_path)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        masks = model.predict(image_tensor)
        mask =fill_holes_and_remove_small_masks(masks[0])
        visualize(image, mask, args.output_path)

    elif args.mode == 'eval':
        evaluate(model, args.data_dir, device)

if __name__ == "__main__":
    main()