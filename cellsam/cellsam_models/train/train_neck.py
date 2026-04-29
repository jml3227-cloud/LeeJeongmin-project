import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cellsam_models.train.dataset import MoNuSACDataset, TNBCDataset, NuInsSegDataset, DeepBacsDataset, collate_fn
from cellsam_models.AnchorDETR.transform import RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90, Compose

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cellfinder_checkpoint', type=str, required=True,
                        help='/workspace/LeeJeongmin-project/cellsam/outputs')
    parser.add_argument('--sam_checkpoint', type=str, default='/workspace/sam_vit_b_01ec64.pth')
    parser.add_argument('--monusac_dir', type=str, default='/workspace/monusac')
    parser.add_argument('--tnbc_dir', type=str, default='/workspace/tnbc')
    parser.add_argument('--nuinsseg_dir', type=str, default='/workspace/nuinsseg')
    parser.add_argument('--deepbacs_dir', type=str, default='/workspace/deepbacs')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='/workspace/LeeJeongmin-project/cellsam/outputs')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--patience', type=int, default=10)

    return parser

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    pred = pred.view(pred.shape[0], -1)
    target = target.view(target.shape[0], -1).float()

    intersection = (pred * target).sum(dim=1)
    dice = (2 * intersection + 1) / (pred.sum(dim=1) + target.sum(dim=1) + 1)
    return 1 - dice.mean()

def compute_loss(pred_masks, pred_ious, gt_masks):
    best_idx = pred_ious.argmax(dim=1)
    pred_best = pred_masks[torch.arange(len(pred_masks)), best_idx].unsqueeze(1)
    
    bce = F.binary_cross_entropy_with_logits(pred_best, gt_masks.float())
    dice = dice_loss(pred_best, gt_masks)
    return bce + dice

def load_vit_weights_from_cellfinder(sam, cellfinder_ckpt_path, device):
    ckpt = torch.load(cellfinder_ckpt_path, map_location=device)

    state_dict = ckpt.get('model', ckpt)

    vit_weights = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.body.'):
            new_key = k[len('backbone.body.'):]

            vit_weights[new_key] = v

    if len(vit_weights) == 0:
        print("miss backbone.body.")
        return
    
    missing, unexpected = sam.image_encoder.load_state_dict(vit_weights, strict=False)
    print("vit가중치 로드 완료")
    if missing:
        print("missing keys 예시:", missing[:3])

def freeze_except_neck(sam):
    for param in sam.image_encoder.parameters():
        param.requires_grad_(False)

    for param in sam.image_encoder.neck.parameters():
        param.requires_grad_(True)
    
    for param in sam.prompt_encoder.parameters():
        param.requires_grad_(False)

    for param in sam.mask_decoder.parameters():
        param.requires_grad_(False)

    trainable = sum(p.numel() for p in sam.parameters() if p.requires_grad)
    total = sum(p.numel() for p in sam.parameters())
    print(f"[Freeze완료] 학습 파라미터:{trainable:,} / 전체: {total:,}")

def boxes_to_sam_format(boxes_cxcywh, img_size=1024):
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return torch.stack([x1, y1, x2, y2], dim=1)

def train_one_epoch(sam, dataloader, optimizer, device, epoch):
    sam.image_encoder.eval()
    sam.image_encoder.neck.train()
    sam.prompt_encoder.eval()
    sam.mask_decoder.eval()

    total_loss = 0
    for step, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        batch_loss = 0
        for b_idx in range(len(images)):
            image = images[b_idx]
            boxes = targets[b_idx]['boxes'].to(device)
            gt_masks = targets[b_idx]['masks'].to(device)

            N = len(boxes)
            if N == 0:
                continue
            
            image_embedding = sam.image_encoder(image.unsqueeze(0))
            boxes_xyxy = boxes_to_sam_format(boxes, img_size=1024)
            pred_masks_list = []
            pred_ious_list = []

            for i in range(N):
                box = boxes_xyxy[i].unsqueeze(0)
                box_input = box.reshape(1, 2, 2)

                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box_input,
                    masks=None
                )
                low_res_masks, iou_predictions = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True
                )

                masks_upsampled = F.interpolate(
                    low_res_masks, size=(1024, 1024), mode='bilinear',
                    align_corners=False
                )
                pred_masks_list.append(masks_upsampled.squeeze(0))
                pred_ious_list.append(iou_predictions.squeeze(0))

            if len(pred_masks_list) == 0:
                continue

            pred_masks = torch.stack(pred_masks_list)
            pred_ious = torch.stack(pred_ious_list)
            gt_masks_input = gt_masks.unsqueeze(1)
            loss = compute_loss(pred_masks, pred_ious, gt_masks_input)
            batch_loss += loss
                
        if isinstance(batch_loss, int):
            continue

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

        if step % 10 == 0:
            print(f"Epoch {epoch} | Step {step}/{len(dataloader)} | loss: {batch_loss.item():.4f}")

    return total_loss / len(dataloader)

@torch.no_grad()
def validate(sam, dataloader, device):
    sam.eval()
    total_loss = 0

    for images, targets in dataloader:
        images = images.to(device)

        batch_loss = 0
        for b_idx in range(len(images)):
            image = images[b_idx]
            boxes = targets[b_idx]['boxes'].to(device)
            gt_masks = targets[b_idx]['masks'].to(device)

            N = len(boxes)
            if N == 0:
                continue

            image_embedding = sam.image_encoder(image.unsqueeze(0))
            boxes_xyxy = boxes_to_sam_format(boxes, img_size=1024)
 
            pred_masks_list = []
            pred_ious_list = []
 
            for i in range(N):
                box = boxes_xyxy[i].unsqueeze(0)
                box_input = box.reshape(1, 2, 2)
 
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box_input,
                    masks=None
                )
 
                low_res_masks, iou_predictions = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True
                )
 
                masks_upsampled = F.interpolate(
                    low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False
                )
 
                pred_masks_list.append(masks_upsampled.squeeze(0))
                pred_ious_list.append(iou_predictions.squeeze(0))
 
            if len(pred_masks_list) == 0:
                continue
 
            pred_masks = torch.stack(pred_masks_list)
            pred_ious = torch.stack(pred_ious_list)
            gt_masks_input = gt_masks.unsqueeze(1)
 
            loss = compute_loss(pred_masks, pred_ious, gt_masks_input)
            batch_loss += loss
 
        if isinstance(batch_loss, int):
            continue
        total_loss += batch_loss.item()
 
    return total_loss / len(dataloader)
 
 
def main():
    parser = get_args_parser()
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    train_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
    ])

    # 1. SAM 불러오기
    print("SAM 불러오는 중...")
    sam = sam_model_registry['vit_b'](checkpoint=args.sam_checkpoint)
    sam = sam.to(device)
 
    # 2. CellFinder checkpoint에서 ViT 가중치 덮어씌우기
    print("CellFinder ViT 가중치 로드 중...")
    load_vit_weights_from_cellfinder(sam, args.cellfinder_checkpoint, device)
 
    # 3. neck만 학습 가능하게 설정
    freeze_except_neck(sam)
 
    # 4. 데이터셋
    train_datasets = []
    val_datasets = []
 
    if os.path.exists(args.monusac_dir):
        train_datasets.append(MoNuSACDataset(args.monusac_dir, split='train', transform=train_transform))
        val_datasets.append(MoNuSACDataset(args.monusac_dir, split='val'))
        print(f"MoNuSAC 로드됨")
 
    if args.tnbc_dir and os.path.exists(args.tnbc_dir):
        train_datasets.append(TNBCDataset(args.tnbc_dir, split='train', transform=train_transform))
        val_datasets.append(TNBCDataset(args.tnbc_dir, split='val'))
        print(f"TNBC 로드됨")
 
    if args.nuinsseg_dir and os.path.exists(args.nuinsseg_dir):
        train_datasets.append(NuInsSegDataset(args.nuinsseg_dir, split='train', transform=train_transform))
        val_datasets.append(NuInsSegDataset(args.nuinsseg_dir, split='val'))
        print(f"NuInsSeg 로드됨")

    if args.deepbacs_dir and os.path.exists(args.deepbacs_dir):
        train_datasets.append(DeepBacsDataset(args.deepbacs_dir, split='train', transform=train_transform))
        val_datasets.append(DeepBacsDataset(args.deepbacs_dir, split='val'))
        print(f"DeepBacs 로드됨")
    
    if len(train_datasets) == 0:
        raise ValueError("데이터셋 경로를 확인해줘")
 
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
 
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
 
    print(f"train: {len(train_dataset)}장, val: {len(val_dataset)}장")
 
    # 5. optimizer (neck 파라미터만)
    neck_params = [p for p in sam.image_encoder.neck.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(neck_params, lr=args.lr, weight_decay=args.weight_decay)
 
    # cosine lr schedule (논문에서 사용)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
 
    # 6. 학습 루프
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
 
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(sam, train_loader, optimizer, device, epoch)
        val_loss = validate(sam, val_loader, device)
        scheduler.step()
 
        print(f"[Epoch {epoch}/{args.epochs}] train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}")
 
        # checkpoint 저장
        ckpt = {
            'epoch': epoch,
            'neck_state_dict': sam.image_encoder.neck.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(ckpt, os.path.join(args.output_dir, 'neck_checkpoint_last.pth'))
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(ckpt, os.path.join(args.output_dir, 'neck_checkpoint_best.pth'))
            print(f"  → best val_loss 갱신: {best_val_loss:.4f} (epoch {epoch})")
        else:
            patience_counter += 1
            print(f"  → patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
 
 
if __name__ == '__main__':
    main()
                
            
