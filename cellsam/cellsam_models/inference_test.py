import tifffile
import numpy as np
from cellSAM import segment_cellular_image, get_model
import matplotlib.pyplot as plt

model = get_model()

img = tifffile.imread("/home/jml3227/MoNuSAC_images_and_annotations/TCGA-5P-A9K0-01Z-00-DX1/TCGA-5P-A9K0-01Z-00-DX1_1.tif")
img = img[:, :, :3]

# img = img[:512, :512, :]
# print("이미지 shape:", img.shape)

mask, _, _ = segment_cellular_image(img, model=model, device='cpu')
# print("마스크 shape:", mask.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mask, cmap='tab20')
axes[1].set_title('Segmentation Mask')
axes[1].axis('off')

plt.savefig('/home/jml3227/cellsam/cellsam_models/result.png')
plt.close()
print("저장완료")