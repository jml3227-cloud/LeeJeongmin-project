import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

from torchvision.transforms.functional import resize, to_pil_image
from sklearn.cluster import KMeans
from segment_anything import sam_model_registry
from cellsam_models.AnchorDETR.models import build_cellfinder

class ResizeLongestSide:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length
    
    def apply_image(self, image: np.ndarray) -> np.ndarray:
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(resize(to_pil_image(image), target_size))
    
    @staticmethod
    def get_preprocess_shape(oldh, oldw, long_side_length):
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        newh = int(newh + 0.5)
        neww = int(neww + 0.5)
        return (newh, neww)
    
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class CellFinder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        args.enc_layers = 6
        args.dec_layers = 6
        args.dim_feedforward = 1024
        args.hidden_dim = 256
        args.dropout = 0.0
        args.nheads = 8
        args.num_query_position = 3500
        args.num_query_pattern = 1
        args.spatial_prior = "learned"
        args.attention_type = "RCDA"
        args.num_feature_levels = 1
        args.device = "cuda"
        args.num_classes = 2

        # additional parameters
        args.only_neck = False
        args.freeze_backbone = False
        args.aux_loss = True
        args.masks = False
        args.cls_loss_coef = 2.0
        args.bbox_loss_coef = 5.0
        args.giou_loss_coef = 2.0
        args.focal_alpha = 0.25
        args.set_cost_class = 2.0
        args.set_cost_bbox = 5.0
        args.set_cost_giou = 2.0
        args.lr_backbone = 0
        args.sam_checkpoint = '/workspace/sam_vit_b_01ec64.pth'

        self.model, _, self.postprocessors = build_cellfinder(args)

    def forward_inference(self, imgs):
        outputs = self.model(imgs)

        orig_target_sizes = torch.stack([torch.tensor(img.shape[-2:]) for img in imgs]).to(imgs.device)
        res = self.postprocessors["bbox"](outputs, orig_target_sizes)
        return res
    
class CellSAM(nn.Module):
    def __init__(self, sam_checkpoint, cellfinder_checkpoint, neck_checkpoint, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.bbox_threshold = 0.4
        self.iou_threshold = 0.5
        self.mask_threshold = 0.5
        self.sam_transform = ResizeLongestSide(1024)

        # Load SAM 
        self.sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
        self.sam = self.sam.to(self.device)

        # Load Cellfinder 
        self.cellfinder = CellFinder(Namespace())
        ckpt = torch.load(cellfinder_checkpoint, map_location=self.device)
        self.cellfinder.model.load_state_dict(ckpt['model'])
        self.cellfinder = self.cellfinder.to(self.device)
        self.cellfinder.eval()

        # load neck wieghts
        neck_ckpt = torch.load(neck_checkpoint, map_location=self.device)
        self.sam.image_encoder.neck.load_state_dict(neck_ckpt['neck_state_dict'])

    def sam_preprocess(self, x: torch.Tensor, return_paddings=False):
        mean = self.sam.pixel_mean.to(x.device)
        std = self.sam.pixel_std.to(x.device)
        x = (x - mean) / std
        h, w = x.shape[-2:]
        padh = self.sam.image_encoder.img_size - h
        padw = self.sam.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        if return_paddings:
            return x, (padh, padw)
        return x
    
    def preprocess_for_cellfinder(self, images):
        imgs = [F.interpolate(img.unsqueeze(0), size=(1024,1024), mode='bilinear',
                              align_corners=False).squeeze(0) for img in images]
        imgs = torch.stack(imgs).to(self.device)
        return imgs
    
    @torch.no_grad()
    def generate_bounding_boxes(self, images):
        imgs = self.preprocess_for_cellfinder(images)
        results = self.cellfinder.forward_inference(imgs)

        boxes_per_image = [r["boxes"] for r in results]
        scores_per_image = [r["scores"] for r in results]

        print(f"bbox 개수 (threshold 전): {len(boxes_per_image[0])}") 

        filtered_boxes = []
        for boxes, scores in zip(boxes_per_image, scores_per_image):
            data = scores.detach().cpu().numpy()
            threshold = self.bbox_threshold
            if len(data) > 1:
                try:
                    kmeans = KMeans(n_clusters=2, random_state=42).fit(data.reshape(-1,1))
                    threshold_cluster = np.mean(kmeans.cluster_centers_)
                    threshold = 0.66 * self.bbox_threshold + 0.33 * threshold_cluster
                except:
                    pass
            
            print(f"scores 최대: {data.max():.4f}, 최소: {data.min():.4f}, 평균: {data.mean():.4f}")
            print(f"threshold: {threshold}")
            filtered_boxes.append(boxes[data > threshold])
            print(f"bbox 개수 (threshold 후): {len(filtered_boxes[-1])}")
        
        return filtered_boxes
    
    @torch.no_grad()
    def generate_embeddings(self, images):
        processed = []
        paddings = []
        for img in images:
            img_np = img.permute(1, 2, 0).cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            img_resized = self.sam_transform.apply_image(img_np)
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().to(self.device)
            img_tensor, padding = self.sam_preprocess(img_tensor, return_paddings=True)
            processed.append(img_tensor)
            paddings.append(padding)

        imgs = torch.stack(processed)
        embeddings = self.sam.image_encoder(imgs)
        return embeddings, paddings
    
    @torch.no_grad()
    def predict(self, images):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()

        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.to(self.device)

        embeddings, paddings = self.generate_embeddings(images)
        boxes_per_image = self.generate_bounding_boxes(images)

        all_masks = []
        for idx in range(len(images)):
            boxes = boxes_per_image[idx]
            if len(boxes) == 0:
                all_masks.append(np.zeros(images[idx].shape[-2:], dtype=np.int32))
                continue

            masks_thresholded = []
            for box in boxes:
                input_box = box.unsqueeze(0).unsqueeze(0)
                sparse_emb, dense_emb = self.sam.prompt_encoder(
                    points=None,
                    boxes=input_box,
                    masks=None
                )
                low_res_masks, iou_pred = self.sam.mask_decoder(
                    image_embeddings=embeddings[idx].unsqueeze(0),
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False
                )

                if iou_pred[0][0] < self.iou_threshold:
                    warnings.warn("Low IOU, ignoring mask.")
                    continue

                low_res_masks = self.sam.postprocess_masks(
                    low_res_masks.cpu(),
                    input_size=torch.tensor([1024 - paddings[idx][0], 1024 - paddings[idx][1]]),
                    original_size=images[idx].shape[-2:]
                )
                mask = (torch.sigmoid(low_res_masks[0, 0]) > self.mask_threshold).numpy().astype(np.uint8)
                mask = mask[:images[idx].shape[-2], :images[idx].shape[-1]]
                masks_thresholded.append(mask)
            
            if masks_thresholded:
                stacked = np.stack(masks_thresholded)
                instance_mask = np.max(
                    stacked * np.arange(1, len(masks_thresholded) + 1)[:, None, None],
                    axis=0
                )
                all_masks.append(instance_mask)
            else:
                all_masks.append(np.zeros(images[idx].shape[-2:], dtype=np.int32))
        
        return all_masks




