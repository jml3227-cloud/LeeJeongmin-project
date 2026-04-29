import random
import torch
import torchvision.transforms.functional as F


def hflip(image, boxes, masks):
    # 이미지 좌우 반전
    image = F.hflip(image)
    
    # boxes: cxcywh 정규화 형식 [cx, cy, w, h]
    # 좌우 반전이면 cx만 바뀜: cx -> 1 - cx
    if len(boxes) > 0:
        boxes = boxes.clone()
        boxes[:, 0] = 1.0 - boxes[:, 0]
    
    # masks 좌우 반전
    if len(masks) > 0:
        masks = masks.flip(-1)
    
    return image, boxes, masks


def vflip(image, boxes, masks):
    # 이미지 상하 반전
    image = F.vflip(image)
    
    # 상하 반전이면 cy만 바뀜: cy -> 1 - cy
    if len(boxes) > 0:
        boxes = boxes.clone()
        boxes[:, 1] = 1.0 - boxes[:, 1]
    
    # masks 상하 반전
    if len(masks) > 0:
        masks = masks.flip(-2)
    
    return image, boxes, masks


def rotate90(image, boxes, masks):
    # 90도 회전 (반시계 방향)
    # F.rotate는 PIL 기준이라 tensor에는 직접 못 씀
    # transpose + flip으로 90도 회전 구현
    image = image.permute(0, 2, 1).flip(-1)  # [C, H, W] -> 90도 회전
    
    if len(boxes) > 0:
        boxes = boxes.clone()
        # cxcywh에서 90도 회전: (cx, cy, w, h) -> (1-cy, cx, h, w)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes[:, 0] = 1.0 - cy
        boxes[:, 1] = cx
        boxes[:, 2] = h
        boxes[:, 3] = w
    
    if len(masks) > 0:
        masks = masks.permute(0, 2, 1).flip(-1)
    
    return image, boxes, masks


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, boxes, masks):
        if random.random() < self.p:
            return hflip(image, boxes, masks)
        return image, boxes, masks


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, boxes, masks):
        if random.random() < self.p:
            return vflip(image, boxes, masks)
        return image, boxes, masks


class RandomRotate90:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image, boxes, masks):
        if random.random() < self.p:
            return rotate90(image, boxes, masks)
        return image, boxes, masks


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, boxes, masks):
        for t in self.transforms:
            image, boxes, masks = t(image, boxes, masks)
        return image, boxes, masks