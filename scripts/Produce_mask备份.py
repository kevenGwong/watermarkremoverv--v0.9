# ----------------- 第1步：导入库 & 加载模型 -----------------
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ 参数设置 ------------------
img_path = "/home/duolaameng/SAM_Remove/Watermark_sam/watermark-segmentation/test/IMG_0095-4.jpg"
ckpt_path = "/home/duolaameng/SAM_Remove/Watermark_sam/output/checkpoints/epoch=071-valid_iou=0.7267.ckpt"
output_dir = os.path.dirname(img_path)

# ------------------ 与训练时相同的归一化参数 ------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ------------------ 与训练时相同的验证数据增强 ------------------
aug_val = A.Compose([
    A.LongestMaxSize(max_size=768),
    A.PadIfNeeded(min_height=768, min_width=768, border_mode=cv2.BORDER_CONSTANT),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
    ToTensorV2(),
])

# ----------------- 与训练时相同的模型定义 -----------------
class WMModel(torch.nn.Module):
    def __init__(self, freeze_encoder=True):
        super().__init__()
        self.net = smp.create_model("FPN", encoder_name="mit_b5", in_channels=3, classes=1)
        
        # 加载预训练权重（如果需要的话）
        # PRETRAIN_W = "/home/duolaameng/SAM_Remove/Watermark_sam/watermark-segmentation/best_watermark_model_mit_b5_best.pth"
        # state = torch.load(PRETRAIN_W, map_location="cpu")
        # self.net.load_state_dict(state, strict=False)

        if freeze_encoder:
            for p in self.net.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.net(x)

# ------------------ 加载图像 ------------------
image = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
orig_h, orig_w = image_rgb.shape[:2]
print(f"🖼️ 原图尺寸: {orig_w} x {orig_h}")

# ---------- 推理前：算好缩放 & padding ----------
scale = min(768 / orig_h, 768 / orig_w)
new_h, new_w = int(orig_h * scale), int(orig_w * scale)
pad_top, pad_left = (768 - new_h) // 2, (768 - new_w) // 2
print(f"📐 缩放比例: {scale:.3f}, 缩放后尺寸: {new_w} x {new_h}")
print(f"📦 Padding: top={pad_top}, left={pad_left}")

# ------------------ 使用与训练时相同的预处理 ------------------
sample = aug_val(image=image_rgb, mask=None)
img_tensor = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]
print(f"📦 预处理后张量尺寸: {img_tensor.shape}")

# ----------------- 构建模型并加载权重 -----------------
model = WMModel(freeze_encoder=False).to(device)

# 加载Lightning权重
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
model.net.load_state_dict(state_dict)
model.eval()  # 关闭Dropout和BatchNorm的训练行为
print(f"✅ 成功加载模型权重: {ckpt_path}")

# ----------------- 推理并输出mask -----------------
with torch.no_grad():
    pred_mask = model(img_tensor)
    pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

print(f"🔍 预测mask尺寸: {pred_mask.shape}")

# ---------- 推理后：先裁 pad，再缩放 ----------
print(f"✂️ 裁剪padding区域: [{pad_top}:{pad_top+new_h}, {pad_left}:{pad_left+new_w}]")
pred_crop = pred_mask[pad_top:pad_top+new_h, pad_left:pad_left+new_w]
print(f"📐 裁剪后尺寸: {pred_crop.shape}")

# 等比例缩放回原图尺寸
pred_mask = cv2.resize(pred_crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
print(f"✅ 最终mask尺寸: {pred_mask.shape}")

# 二值化 & 膨胀
binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

# 保存结果
base_name = os.path.splitext(os.path.basename(img_path))[0]
binary_mask_path = os.path.join(output_dir, f"{base_name}_binary_mask.png")
dilated_mask_path = os.path.join(output_dir, f"{base_name}_dilated_mask.png")
raw_mask_path = os.path.join(output_dir, f"{base_name}_raw_mask.png")

cv2.imwrite(binary_mask_path, binary_mask)
cv2.imwrite(dilated_mask_path, dilated_mask)
cv2.imwrite(raw_mask_path, (pred_mask * 255).astype(np.uint8))

print(f"✅ Saved: {binary_mask_path}")
print(f"✅ Saved: {dilated_mask_path}")
print(f"✅ Saved: {raw_mask_path}")

# 显示结果
plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(pred_mask, cmap='gray')
plt.title("Raw Prediction")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(binary_mask, cmap='gray')
plt.title("Binary Mask")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(dilated_mask, cmap='gray')
plt.title("Dilated Mask")
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_name}_results.png"), dpi=150, bbox_inches='tight')
plt.show()
