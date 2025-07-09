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
from typing import List, Tuple

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ 参数设置 ------------------
img_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/IMG_0242-3.jpg"
ckpt_path = "/home/duolaameng/SAM_Remove/WatermarkRemover-AI/data/models/epoch=136-valid_iou=0.8730.ckpt"
output_dir = os.path.dirname(img_path)

# ------------------ 与训练时相同的归一化参数 ------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ------------------ 与训练时相同的验证数据增强（512分辨率） ------------------
aug_val = A.Compose([
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

def get_padding(h: int, w: int,
                crop: int = 512,
                step: int = 256) -> Tuple[int, int, int, int]:
    """
    计算在 (top, bottom, left, right) 四个方向各补多少像素，
    使得 padded_h、padded_w 能以 `step` 步长完整覆盖。
    """
    # 保证滑窗能走到最右 / 最下而不欠像素
    need_h = (np.ceil((h - crop) / step) * step + crop).astype(int)
    need_w = (np.ceil((w - crop) / step) * step + crop).astype(int)
    pad_h = need_h - h
    pad_w = need_w - w
    # 上下左右各分一半（奇数时多补到 bottom / right）
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    return top, bottom, left, right

def infer_full_image(img_bgr: np.ndarray,
                     model: torch.nn.Module,
                     device: torch.device,
                     crop: int = 512,
                     step: int = 256) -> np.ndarray:
    """
    输入 BGR 原图，输出与原图尺寸一致的 float mask (0~1)
    —— 完全等同训练时代码的切块 / padding 逻辑
    """
    h0, w0 = img_bgr.shape[:2]

    # ---------- ① 先整体 padding ----------
    top, bottom, left, right = get_padding(h0, w0, crop, step)
    img_pad = cv2.copyMakeBorder(
        img_bgr, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    H, W = img_pad.shape[:2]
    out_pad = np.zeros((H, W), dtype=np.float32)
    cnt_pad = np.zeros((H, W), dtype=np.float32)

    # ---------- ② 滑窗切块推理 ----------
    for y in range(0, H - crop + 1, step):
        for x in range(0, W - crop + 1, step):
            patch = img_pad[y:y+crop, x:x+crop]
            sample = aug_val(image=patch)        # 与训练同样的 aug_val
            inp = sample["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

            out_pad[y:y+crop, x:x+crop] += pred
            cnt_pad[y:y+crop, x:x+crop] += 1.0

    out_pad /= cnt_pad  # 重叠区做平均

    # ---------- ③ 把 padding 裁掉，恢复原尺寸 ----------
    full_mask = out_pad[top:top+h0, left:left+w0]
    return full_mask

# ------------------ 加载图像 ------------------
image = cv2.imread(img_path)
orig_h, orig_w = image.shape[:2]
print(f"🖼️ 原图尺寸: {orig_w} x {orig_h}")

# ----------------- 构建模型并加载权重 -----------------
model = WMModel(freeze_encoder=False).to(device)

# 加载Lightning权重
ckpt = torch.load(ckpt_path, map_location=device)
state_dict = {k.replace("net.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
model.net.load_state_dict(state_dict)
model.eval()  # 关闭Dropout和BatchNorm的训练行为
print(f"✅ 成功加载模型权重: {ckpt_path}")

# ----------------- 使用完全一致的推理逻辑 -----------------
print("🔍 开始推理...")
pred_mask = infer_full_image(image, model, device)
print(f"✅ 推理完成，mask尺寸: {pred_mask.shape}")

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
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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
