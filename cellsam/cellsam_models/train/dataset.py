import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import resize, InterpolationMode


class MoNuSACDataset(Dataset):
    def __init__(self, root_dir):
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tif_path, xml_path = self.samples[idx]

        image = np.array(Image.open(tif_path).convert('RGB'))
        H, W = image.shape[:2]

        boxes, masks = self.parse_xml(xml_path, H, W)

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)

        image = resize(image, [1024, 1024])

        scale_x = 1024 / W
        scale_y = 1024 / H
        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y

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
            boxes.append([x_min, y_min, x_max, y_max])

            # binary mask
            mask = np.zeros((H,W), dtype=np.uint8)
            rr, cc = sk_polygon(ys, xs, shape=(H,W))
            mask[rr, cc] = 1
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

if __name__ == '__main__':
    dataset = MoNuSACDataset('/home/jml3227/MoNuSAC_processed')

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    images, targets = batch
    print(f'배치 이미지 shape: {images.shape}')
    print(f'targets 개수: {len(targets)}')
    print(f'targets[0] keys: {targets[0].keys()}')
    print(f'boxes shape: {targets[0]["boxes"].shape}')
    print(f'labels shape: {targets[0]["labels"].shape}')