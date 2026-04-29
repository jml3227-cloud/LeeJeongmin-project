# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AnchorDETR'))

import torch
import torch.nn.functional as F
from torch import nn
from segment_anything import sam_model_registry
from ..util.misc import NestedTensor, is_main_process


class ModifiedImageEncoderViT(nn.Module):
    def __init__(self, original_model):
        super(ModifiedImageEncoderViT, self).__init__()

        self.patch_embed = original_model.patch_embed
        self.blocks = original_model.blocks
        self.pos_embed = original_model.pos_embed

    def forward(self, x):
        x = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        
        return x.permute(0, 3, 1, 2)
    

class SAMBackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        only_neck: bool,
        freeze_backbone: bool = False,
        sam_vit: str = "vit_b"      # 논문에서는 vit-b로 했으나 update된 깃허브에는 vit-h임
    ):
        super().__init__()
        if freeze_backbone:
            for name, parameters in backbone.named_parameters():
                parameters.requires_grad_(False)
        if only_neck:
            for name, parameters in backbone.named_parameters():
                if "neck" in name:
                    parameters.requires_grad_(True)
                else:
                    parameters.requires_grad_(False)
        
        self.strides = [16]
        self.num_channels = [768]
        self.body = backbone

    def forward(self, tensor_list: NestedTensor):
        x = self.body(tensor_list.tensors)
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out = NestedTensor(x, mask)
        return [out]
    
class SAMBackbone(SAMBackboneBase):
    def __init__(
        self,
        train_backbone: bool,
        only_neck: bool = False,
        freeze_backbone: bool = False,
        sam_vit: str = "vit_b",
        sam_checkpoint: str = None,
    ):
        backbone = sam_model_registry[sam_vit](checkpoint=sam_checkpoint)
        backbone = backbone.image_encoder
        backbone = ModifiedImageEncoderViT(backbone)
        super().__init__(
            backbone,
            train_backbone,
            only_neck,
            freeze_backbone,
            sam_vit,
        )


def build_backbone(args):
    backbone = SAMBackbone(
        train_backbone=args.lr_backbone > 0,
        only_neck=args.only_neck,
        freeze_backbone=args.freeze_backbone,
        sam_vit="vit_b",
        sam_checkpoint=args.sam_checkpoint
    )
    return backbone
