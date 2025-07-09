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

# ------------------ 全图推理的数据增强（resize到512x512） ------------------
aug_full = A.Compose([
    A.Resize(512, 512),
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

def infer_full_image_less_overlap(img_bgr: np.ndarray,
                                 model: torch.nn.Module,
                                 device: torch.device,
                                 crop: int = 512,
                                 step: int = 384) -> np.ndarray:
    """
    减少重叠面积的切块推理（步长384，重叠约25%）
    """
    h0, w0 = img_bgr.shape[:2]

    # ---------- ① 先整体 padding ----------
    top, bottom, left, right = get_padding(h0, w0, crop, step)
    img_pad = cv2.copyMakeBorder(
        img_bgr, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    H, W = img_pad.shape[:2]
    out_pad = np.zeros((H, W), dtype=np.float32)
    cnt_pad = np.zeros((H, W), dtype=np.float32)

    # ---------- ② 滑窗切块推理（减少重叠） ----------
    for y in range(0, H - crop + 1, step):
        for x in range(0, W - crop + 1, step):
            patch = img_pad[y:y+crop, x:x+crop]
            sample = aug_val(image=patch)
            inp = sample["image"].unsqueeze(0).to(device)

            with torch.no_grad():
                pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

            out_pad[y:y+crop, x:x+crop] += pred
            cnt_pad[y:y+crop, x:x+crop] += 1.0

    out_pad /= cnt_pad  # 重叠区做平均

    # ---------- ③ 把 padding 裁掉，恢复原尺寸 ----------
    full_mask = out_pad[top:top+h0, left:left+w0]
    return full_mask

def infer_full_image_resize(img_bgr: np.ndarray,
                           model: torch.nn.Module,
                           device: torch.device) -> np.ndarray:
    """
    全图resize到512x512进行推理，然后resize回原尺寸
    """
    h0, w0 = img_bgr.shape[:2]
    
    # 预处理：resize到512x512
    sample = aug_full(image=img_bgr)
    inp = sample["image"].unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
    
    # resize回原尺寸
    pred_resized = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_LINEAR)
    
    return pred_resized

def count_patches(h: int, w: int, crop: int = 512, step: int = 256) -> int:
    """计算切块数量"""
    H = h + (crop - h % step) if h % step != 0 else h
    W = w + (crop - w % step) if w % step != 0 else w
    num_h = (H - crop) // step + 1
    num_w = (W - crop) // step + 1
    return num_h * num_w

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

# ----------------- 计算切块数量对比 -----------------
patches_256 = count_patches(orig_h, orig_w, 512, 256)
patches_384 = count_patches(orig_h, orig_w, 512, 384)
print(f"📊 切块数量对比:")
print(f"   步长256（50%重叠）: {patches_256} 个切块")
print(f"   步长384（25%重叠）: {patches_384} 个切块")
print(f"   减少比例: {(patches_256 - patches_384) / patches_256 * 100:.1f}%")

# ----------------- 方法1：使用完全一致的切块推理逻辑 -----------------
print("\n🔍 方法1：开始切块推理（步长256，50%重叠）...")
pred_mask_patch = infer_full_image(image, model, device)
print(f"✅ 切块推理完成，mask尺寸: {pred_mask_patch.shape}")

# ----------------- 方法2：减少重叠的切块推理 -----------------
print("🔍 方法2：开始减少重叠切块推理（步长384，25%重叠）...")
pred_mask_less = infer_full_image_less_overlap(image, model, device)
print(f"✅ 减少重叠推理完成，mask尺寸: {pred_mask_less.shape}")

# ----------------- 方法3：全图resize推理 -----------------
print("🔍 方法3：开始全图推理...")
pred_mask_full = infer_full_image_resize(image, model, device)
print(f"✅ 全图推理完成，mask尺寸: {pred_mask_full.shape}")

# ----------------- 保存三种方法的结果 -----------------
base_name = os.path.splitext(os.path.basename(img_path))[0]

# 切块推理结果（步长256）
binary_mask_patch = (pred_mask_patch > 0.5).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilated_mask_patch = cv2.dilate(binary_mask_patch, kernel, iterations=1)

patch_binary_path = os.path.join(output_dir, f"{base_name}_patch_binary_mask.png")
patch_dilated_path = os.path.join(output_dir, f"{base_name}_patch_dilated_mask.png")
patch_raw_path = os.path.join(output_dir, f"{base_name}_patch_raw_mask.png")

cv2.imwrite(patch_binary_path, binary_mask_patch)
cv2.imwrite(patch_dilated_path, dilated_mask_patch)
cv2.imwrite(patch_raw_path, (pred_mask_patch * 255).astype(np.uint8))

# 减少重叠切块推理结果（步长384）
binary_mask_less = (pred_mask_less > 0.5).astype(np.uint8) * 255
dilated_mask_less = cv2.dilate(binary_mask_less, kernel, iterations=1)

less_binary_path = os.path.join(output_dir, f"{base_name}_less_binary_mask.png")
less_dilated_path = os.path.join(output_dir, f"{base_name}_less_dilated_mask.png")
less_raw_path = os.path.join(output_dir, f"{base_name}_less_raw_mask.png")

cv2.imwrite(less_binary_path, binary_mask_less)
cv2.imwrite(less_dilated_path, dilated_mask_less)
cv2.imwrite(less_raw_path, (pred_mask_less * 255).astype(np.uint8))

# 全图推理结果
binary_mask_full = (pred_mask_full > 0.5).astype(np.uint8) * 255
dilated_mask_full = cv2.dilate(binary_mask_full, kernel, iterations=1)

full_binary_path = os.path.join(output_dir, f"{base_name}_full_binary_mask.png")
full_dilated_path = os.path.join(output_dir, f"{base_name}_full_dilated_mask.png")
full_raw_path = os.path.join(output_dir, f"{base_name}_full_raw_mask.png")

cv2.imwrite(full_binary_path, binary_mask_full)
cv2.imwrite(full_dilated_path, dilated_mask_full)
cv2.imwrite(full_raw_path, (pred_mask_full * 255).astype(np.uint8))

print(f"✅ 切块推理结果保存（步长256）:")
print(f"   - {patch_binary_path}")
print(f"   - {patch_dilated_path}")
print(f"   - {patch_raw_path}")
print(f"✅ 减少重叠推理结果保存（步长384）:")
print(f"   - {less_binary_path}")
print(f"   - {less_dilated_path}")
print(f"   - {less_raw_path}")
print(f"✅ 全图推理结果保存:")
print(f"   - {full_binary_path}")
print(f"   - {full_dilated_path}")
print(f"   - {full_raw_path}")

# ----------------- 对比显示结果 -----------------
plt.figure(figsize=(24, 12))

# 原图
plt.subplot(3, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

# 切块推理结果（步长256）
plt.subplot(3, 4, 2)
plt.imshow(pred_mask_patch, cmap='gray')
plt.title("Patch Inference (step=256) - Raw")
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(binary_mask_patch, cmap='gray')
plt.title("Patch Inference (step=256) - Binary")
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(dilated_mask_patch, cmap='gray')
plt.title("Patch Inference (step=256) - Dilated")
plt.axis('off')

# 减少重叠推理结果（步长384）
plt.subplot(3, 4, 5)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(3, 4, 6)
plt.imshow(pred_mask_less, cmap='gray')
plt.title("Patch Inference (step=384) - Raw")
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(binary_mask_less, cmap='gray')
plt.title("Patch Inference (step=384) - Binary")
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(dilated_mask_less, cmap='gray')
plt.title("Patch Inference (step=384) - Dilated")
plt.axis('off')

# 全图推理结果
plt.subplot(3, 4, 9)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(3, 4, 10)
plt.imshow(pred_mask_full, cmap='gray')
plt.title("Full Image Inference - Raw")
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(binary_mask_full, cmap='gray')
plt.title("Full Image Inference - Binary")
plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(dilated_mask_full, cmap='gray')
plt.title("Full Image Inference - Dilated")
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{base_name}_three_methods_comparison.png"), dpi=150, bbox_inches='tight')
plt.show()

# ----------------- 计算差异统计 -----------------
diff_mask_256_384 = np.abs(pred_mask_patch - pred_mask_less)
diff_binary_256_384 = np.abs(binary_mask_patch - binary_mask_less)

diff_mask_256_full = np.abs(pred_mask_patch - pred_mask_full)
diff_binary_256_full = np.abs(binary_mask_patch - binary_mask_full)

print(f"\n📊 三种方法差异统计:")
print(f"   步长256 vs 步长384:")
print(f"     原始mask平均差异: {np.mean(diff_mask_256_384):.4f}")
print(f"     二值化mask差异像素数: {np.sum(diff_binary_256_384 > 0)}")
print(f"     二值化mask差异比例: {np.sum(diff_binary_256_384 > 0) / (orig_h * orig_w) * 100:.2f}%")
print(f"   步长256 vs 全图推理:")
print(f"     原始mask平均差异: {np.mean(diff_mask_256_full):.4f}")
print(f"     二值化mask差异像素数: {np.sum(diff_binary_256_full > 0)}")
print(f"     二值化mask差异比例: {np.sum(diff_binary_256_full > 0) / (orig_h * orig_w) * 100:.2f}%")
