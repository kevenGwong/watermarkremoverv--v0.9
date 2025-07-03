#!/usr/bin/env python3
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor

# === 参数 ===
input_dir  = Path("/home/duolaameng/SAM_Remove/Watermark_sam/input")
output_dir = Path("/hom e/duolaameng/SAM_Remove/Watermark_sam/clean")
mask_path  = Path("/home/duolaameng/SAM_Remove/WatermarkRemover-AI/test.jpg")
max_parallel = 2

# === 收集已处理记录（文件 & 子文件夹都算） ===
processed = set()
for p in output_dir.rglob("*"):
    if p.is_file() or (p.is_dir() and p.suffix):  # 目录名里带扩展名的情况
        processed.add(p.stem.lower())

# === 获取待处理列表 ===
img_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
imglist = [p for ext in img_exts for p in input_dir.glob(ext)]

print(f"共检测到 {len(imglist)} 张图片")

def process(img_path: Path):
    stem = img_path.stem.lower()
    if stem in processed:
        print(f"已存在：{stem}，跳过")
        return

    out_path = output_dir / f"{stem}.png"   # 指定“文件”而非目录
    cmd = [
        "conda", "run", "-n", "py311aiwatermark", "iopaint", "run",
        "--image",  str(img_path),
        "--mask",   str(mask_path),
        "--output", str(out_path),
        "--model",  "lama",
        "--device", "cuda",
    ]
    print(f"处理：{img_path.name} → {out_path.name}")
    subprocess.run(cmd, check=True)
    processed.add(stem)       # 并发环境下及时登记

# === 并行执行 ===
with ThreadPoolExecutor(max_workers=max_parallel) as pool:
    pool.map(process, imglist)

print("全部处理完成")
