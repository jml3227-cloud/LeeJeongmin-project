import argparse
import torch
from torch.utils.data import DataLoader
import sys
import os

from cellsam_models.anchor_detr import build
from cellsam_models.train.dataset import MoNuSACDataset, collate_fn

def get_args_parser():
    parser = argparse.ArgumentParser('CellSAM Training')
    
    # 학습 관련
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1, type=int)    # 실제 학습시 에포크 수정
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    
    # backbone
    parser.add_argument('--only_neck', default=False, action='store_true')
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--num_feature_levels', default=1, type=int)
    parser.add_argument('--sam_checkpoint', 
                    default='/home/jml3227/cellsam/sam_vit_b_01ec64.pth', 
                    type=str)
    
    # transformer
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
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
    
    # 기타
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--data_path', default='/home/jml3227/MoNuSAC_processed', type=str)
    parser.add_argument('--output_dir', default='/home/jml3227/cellsam/outputs', type=str)
    
    return parser

def main(args):
    device = torch.device(args.device)

    # 모델 build
    model, criterion, postprocessors = build(args)
    model.to(device)
    criterion.to(device)

    # dataset, dataloader
    dataset = MoNuSACDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate_fn)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    for epoch in range(args.epochs):
        model.train()
        criterion.train()

        for i, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward
            outputs = model(images)

            # loss 계산
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k]
                         for k in loss_dict.keys() if k in weight_dict)
            
            # backward 
            optimizer.zero_grad()
            losses.backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch+1} / {args.epochs}]'
                      f'Step [{i+1} / {len(dataloader)}]'
                      f'Loss: {losses.item():.4f}')
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch{epoch}.pth'))
                
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)