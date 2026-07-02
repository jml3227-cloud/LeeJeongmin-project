import os
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon
from skimage.measure import label, regionprops
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, InterpolationMode

class CellSAMFullNpyDataset(Dataset):
    SUBSETS = [
        '2b_brightfield_dataset',
        '2b_fluorescence_dataset',
        '2c_e_coli',
        '2d_1_SplineDist_dataset',
        '2d_2_b_subtilis',
        '2e_e_coli',
        '3T3_ep_microscopy',
        'A549_ep_microscopy',
        'CHO_ep_microscopy',
        'Gendarme_BriFi',
        'HEK293_ep_microscopy',
        'HeLa-S3_ep_microscopy',
        'HeLa_ep_microscopy',
        'PC3_ep_microscopy',
        'RAW264_ep_microscopy',
        'YeaZ',
        'YeastNet',
        'bact_fluor',
        'bact_phase',
        'cellpose',
        'cpm15',
        'cpm17',
        'dsb_fixed',
        'kumar',
        'monusac',
        'monuseg',
        'nuinsseg',
        's2_stardist',
        'tissuenet_nuclear',
        'tissuenet_wholecell',
        'tnbc',
    ]

    def __init__(self, root_dir, split='train', transform=None):
        self.split = split
        self.transform = transform
        self.samples = []
        self.dataset_size = []

        split_dir = os.path.join(root_dir, split)

        for subset in self.SUBSETS:
            subset_dir = os.path.join(split_dir, subset)
            if not os.path.isdir(subset_dir):
                self.dataset_size.append(0)
                continue

            files = os.listdir(subset_dir)
            x_files = sorted([f for f in files if f.endswith('.X.npy')])

            subset_samples = []
            for x_file in x_files:
                base = x_file[:-len('.X.npy')]
                y_file = base + '.y.npy'
                if y_file in files:
                    subset_samples.append((
                        os.path.join(subset_dir, x_file),
                        os.path.join(subset_dir, y_file)
                    ))
            self.samples.extend(subset_samples)
            self.dataset_size.append(len(subset_samples))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = np.load(img_path)
        gt = np.load(mask_path).squeeze(0)

        H, W = image.shape[1:]

        boxes, masks = self.parse_mask(gt)

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self.samples))
        
        image = torch.tensor(image, dtype=torch.float32)
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

        if self.transform is not None and self.split == 'train':
            image, boxes, masks = self.transform(image, boxes, masks)

        return image, boxes, masks
    
    def parse_mask(self, gt):
        boxes = []
        masks = []

        for region in regionprops(gt):
            if region.area < 10:
                continue

            y_min, x_min, y_max, x_max = region.bbox
            boxes.append([x_min, y_min, x_max, y_max])
            mask = (gt == region.label).astype(np.uint8)
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
