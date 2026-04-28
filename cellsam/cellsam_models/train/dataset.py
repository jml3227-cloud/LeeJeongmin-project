import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon
from skimage.measure import label, regionprops
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, InterpolationMode
from sklearn.model_selection import train_test_split


class MoNuSACDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.samples = []
        files = os.listdir(root_dir)
        tif_files = sorted([f for f in files if f.endswith('.tif')])

        for tif_file in tif_files:
            base = tif_file.replace('.tif', '')
            xml_file = base + '.xml'
            if xml_file in files:
                self.samples.append((
                    os.path.join(root_dir, tif_file),
                    os.path.join(root_dir, xml_file)
                ))

        train_samples, temp_samples = train_test_split(self.samples, test_size=0.2, random_state=42)
        val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)

        if split == 'train':
            self.samples = train_samples
        elif split == 'val':
            self.samples = val_samples
        elif split == 'test':
            self.samples = test_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tif_path, xml_path = self.samples[idx]

        image = np.array(Image.open(tif_path).convert('RGB'))
        H, W = image.shape[:2]

        boxes, masks = self.parse_xml(xml_path, H, W)

        if len(boxes) > 450:
            return self.__getitem__((idx + 1) % len(self.samples))
    
        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.samples))

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)

        image = resize(image, [1024, 1024])

        x_min = boxes[:, 0] / W
        y_min = boxes[:, 1] / H
        x_max = boxes[:, 2] / W
        y_max = boxes[:, 3] / H

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        boxes = torch.stack([cx, cy, w, h], dim=1)

        masks = resize(masks, [1024, 1024], interpolation=InterpolationMode.NEAREST)

        return image, boxes, masks
    
    def parse_xml(self, xml_path, H, W):
        tree = ET.parse(xml_path)       # xml 파일 읽기
        root = tree.getroot()     # 최상위 노드(root) 가져오기

        boxes = []
        masks =[]

        for region in root.iter('Region'):
            vertices = region.find('Vertices')
            if vertices is None:
                continue

            coords = [(int(v.get('X')), int(v.get('Y')))
                      for v in vertices.findall('Vertex')]
            if len(coords) < 3:
                continue

            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]

            # bounding box
            x_min = max(0, min(xs))
            x_max = min(W-1, max(xs))
            y_min = max(0, min(ys))
            y_max = min(H-1, max(ys))

            if x_max <= x_min or y_max <= y_min:
                continue

            boxes.append([x_min, y_min, x_max, y_max])

            # binary mask
            mask = np.zeros((H,W), dtype=np.uint8)
            rr, cc = sk_polygon(ys, xs, shape=(H,W))
            mask[rr, cc] = 1
            masks.append(mask)

        return boxes, masks
    
class TNBCDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.samples = []

        files = os.listdir(root_dir)
        img_files = sorted([f for f in files if f.endswith('.png') and not f.startswith('GT_')])

        for img_file in img_files:
            gt_file = 'GT_' + img_file
            if gt_file in files:
                self.samples.append((
                    os.path.join(root_dir, img_file),
                    os.path.join(root_dir, gt_file)
                ))

        train_samples, temp_samples = train_test_split(self.samples, test_size=0.2, random_state=42)
        val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)

        if split == 'train':
            self.samples = train_samples
        elif split == 'val':
            self.samples = val_samples
        elif split == 'test':
            self.samples = test_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, gt_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert('RGB'))
        gt = np.array(Image.open(gt_path).convert('L'))
        H, W = image.shape[:2]

        boxes, masks = self.parse_mask(gt, H, W)

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.samples))
            
        if len(boxes) > 450:
            return self.__getitem__((idx + 1) % len(self.samples))
        
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)

        image = resize(image, [1024, 1024])

        x_min = boxes[:, 0] / W
        y_min = boxes[:, 1] / H
        x_max = boxes[:, 2] / W
        y_max = boxes[:, 3] / H

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        boxes = torch.stack([cx, cy, w, h], dim=1)

        masks = resize(masks, [1024, 1024], interpolation=InterpolationMode.NEAREST)

        return image, boxes, masks
    
    def parse_mask(self, gt, H, W):
        binary = (gt > 0).astype(np.uint8)
        labeled = label(binary)

        boxes = []
        masks = []

        for region in regionprops(labeled):
            if region.area < 10:
                continue

            y_min, x_min, y_max, x_max = region.bbox

            boxes.append([x_min, y_min, x_max, y_max])
            mask = (labeled == region.label).astype(np.uint8)
            masks.append(mask)

        return boxes, masks
    
class NuInsSegDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.samples = []

        files = os.listdir(root_dir)
        img_files = sorted([f for f in files if f.startswith('image_') and f.endswith('.png')])
        
        for img_file in img_files:
            mask_file = 'mask_' + img_file[6:]
            if mask_file in files:
                self.samples.append((
                    os.path.join(root_dir, img_file),
                    os.path.join(root_dir, mask_file)
                ))

        train_samples, temp_samples = train_test_split(self.samples, test_size=0.2, random_state=42)
        val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)

        if split == 'train':
            self.samples = train_samples
        elif split == 'val':
            self.samples = val_samples
        elif split == 'test':
            self.samples = test_samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert('RGB'))
        gt = np.array(Image.open(mask_path).convert('L'))
        H, W = image.shape[:2]

        boxes, masks = self.parse_mask(gt, H, W)

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.samples))
        
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)

        image = resize(image, [1024, 1024])

        x_min = boxes[:, 0] / W
        y_min = boxes[:, 1] / H
        x_max = boxes[:, 2] / W
        y_max = boxes[:, 3] / H

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min)
        h = (y_max - y_min)

        boxes = torch.stack([cx, cy, w, h], dim=1)

        masks = resize(masks, [1024, 1024], interpolation=InterpolationMode.NEAREST)

        return image, boxes, masks

    def parse_mask(self, gt, H, W):
        binary = (gt > 0).astype(np.uint8)
        labeled = label(binary)

        boxes = []
        masks = []

        for region in regionprops(labeled):
            if region.area < 10:
                continue

            y_min, x_min, y_max, x_max = region.bbox

            boxes.append([x_min, y_min, x_max, y_max])
            mask = (labeled == region.label).astype(np.uint8)
            masks.append(mask)

        return boxes, masks

        
    
def collate_fn(batch):
    images, boxes, masks = zip(*batch)
    images = torch.stack(images)

    targets = []
    for box, mask in zip(boxes, masks):
        N = len(box)
        targets.append({
            'boxes': box,
            'masks': mask,
            'labels': torch.zeros(N, dtype=torch.int64)
        })

    return images, targets
