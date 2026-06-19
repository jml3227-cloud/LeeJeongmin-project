import argparse
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import sys
import os

from cellsam_models.AnchorDETR.models import build_cellfinder
from cellsam_models.train.dataset import DeepBacsNpyDataset, collate_fn
from cellsam_models.AnchorDETR.transform import RandomHorizontalFlip, RandomVerticalFlip, RandomRotate90, Compose

def get_args_parser():
    parser = argparse.ArgumentParser('CellSAM Training - DeepBacs Reproduction')
    
    # 학습 관련
    parser.add_argument('--lr', default=1e-4, type=float) 
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--patience', default=5, type=int)  # early stopping patience
    
    # backbone
    parser.add_argument('--only_neck', default=False, action='store_true')
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--num_feature_levels', default=1, type=int)
    parser.add_argument('--sam_checkpoint', 
                    default='/workspace/sam_vit_b_01ec64.pth', 
                    type=str)
    
    # transformer
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--num_query_position', default=3500, type=int)
    parser.add_argument('--num_query_pattern', default=1, type=int)
    parser.add_argument('--spatial_prior', default='learned', type=str)
    parser.add_argument('--attention_type', default='RCDA', type=str)
    
    # loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.add_argument('--masks', default=False, action='store_true')
    parser.add_argument('--set_cost_class', default=2, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    #데이터
    parser.add_argument('--deepbacs_root', default='/workspace/cellsam_v1.2', type=str)
    parser.add_argument('--max_instances', default=400, type=int)

    # 기타
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='/workspace/LeeJeongmin-project/cellsam/outputs', type=str)
    parser.add_argument('--resume', default=None, type=str)
    
    return parser

def main(args):
    device = torch.device(args.device)

    # 모델 build
    model, criterion, postprocessors = build_cellfinder(args)
    model.to(device)
    criterion.to(device)

    train_transform = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotate90(p=0.5)
    ])

    # dataset, dataloader
    train_dataset = DeepBacsNpyDataset(args.deepbacs_root, split='train',
                                       transform=train_transform,
                                       max_instances=args.max_instances)
    
    val_dataset = DeepBacsNpyDataset(args.deepbacs_root, split='val',
                                     max_instances=args.max_instances)
    print(f"train: {len(train_dataset)}장, val: {len(val_dataset)}장")
    print(f"subset별 개수 (train): {dict(zip(train_dataset.SUBSETS, train_dataset.dataset_size))}")

    train_sampler = WeightedRandomSampler(
        train_dataset.get_sample_weights(),
        num_samples=len(train_dataset),
        replacement=True
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  sampler=train_sampler, collate_fn=collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=4)
    
    # optimizer
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': args.lr_backbone},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': args.lr}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # TensorBoard writer
    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensor_board_logs'))

    log_path = os.path.join(args.output_dir, 'train_log.txt')

    # early stopping 변수
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_val_loss = float('inf')
        print(f'resume: epoch {start_epoch}부터 시작')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        criterion.train()

        epoch_train_loss = 0.0
        num_steps = 0

        for i, (images, targets) in enumerate(train_dataloader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k]
                         for k in loss_dict.keys() if k in weight_dict)
            
            optimizer.zero_grad()
            losses.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            epoch_train_loss += losses.item()
            num_steps += 1

            if i % 10 == 0:
                print(f'Epoch [{epoch+1}/{args.epochs}] Step [{i+1}/{len(train_dataloader)}] '
                    f'loss_ce: {loss_dict["loss_ce"].item():.4f} | '
                    f'loss_bbox: {loss_dict["loss_bbox"].item():.4f} | '
                    f'loss_giou: {loss_dict["loss_giou"].item():.4f} | '
                    f'total: {losses.item():.4f}')
        
        epoch_train_loss /= num_steps
            
        # validation
        model.eval()
        criterion.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_dataloader:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                losses = sum(loss_dict[k] * weight_dict[k]
                         for k in loss_dict.keys() if k in weight_dict)
                print(f"loss: {losses.item():.4f}, boxes: {[len(t['boxes']) for t in targets]}")
                val_loss += losses.item()
        
        val_loss /= len(val_dataloader)
        print(f'Epoch [{epoch+1} / {args.epochs}] Val loss: {val_loss:.4f}')

        writer.add_scalars('Loss', {'train': epoch_train_loss, 'val': val_loss}, epoch + 1)

        # checkpoint 저장
        ckpt = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
        }

        # 마지막 epoch 저장
        torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint_last.pth'))

        # best val loss 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(ckpt, os.path.join(args.output_dir, 'checkpoint_best.pth'))
            print(f'  → best 갱신: {best_val_loss:.4f} (epoch {epoch+1})')
        else:
            patience_counter += 1
            print(f'  → patience: {patience_counter}/{args.patience}')
            if patience_counter >= args.patience:
                stop_msg = (
                    f'Early stopping at epoch {epoch+1}\n'
                    f'best_epoch: {best_epoch}\n'
                    f'best_val_loss: {best_val_loss:.4f}\n'
                )
                print(stop_msg)
                with open(log_path, 'a') as f:
                    f.write(stop_msg)
                break
        
    else:
        final_msg = (
            f'Training finished without early stopping (reached --epochs={args.epochs})\n'
            f'best_epoch: {best_epoch}\n'
            f'best_val_loss: {best_val_loss:.4f}\n'
        )
        print(final_msg)
        with open(log_path, 'a') as f:
            f.write(final_msg)

    writer.close()
    
                
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)