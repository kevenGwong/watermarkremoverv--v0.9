"""
CLI界面 - 用于调试和批量处理
"""

import argparse
import logging
from pathlib import Path
from PIL import Image
from typing import Dict, Any
import sys

# 添加包路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from watermark_remover_ai.core.processors.watermark_processor import WatermarkProcessor

logger = logging.getLogger(__name__)

def main(args, config: Dict[str, Any]):
    """CLI主函数"""
    try:
        # 初始化处理器
        processor = WatermarkProcessor(config)
        
        # 处理单个文件
        if not args.batch:
            process_single_file(args, processor)
        else:
            process_directory(args, processor)
            
    except Exception as e:
        logger.error(f"CLI execution failed: {e}")
        sys.exit(1)

def process_single_file(args, processor):
    """处理单个文件"""
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing: {input_path} -> {output_path}")
    
    try:
        # 加载图像
        image = Image.open(input_path)
        
        # 处理参数
        mask_method = args.mask_method if args.mask_method != "auto" else "custom"
        
        # 处理图像
        result = processor.process_image(
            image=image,
            mask_method=mask_method,
            transparent=args.transparent,
            max_bbox_percent=args.max_bbox_percent,
            force_format=args.force_format
        )
        
        if result.success:
            # 保存结果
            result.result_image.save(output_path)
            logger.info(f"✅ Successfully saved: {output_path}")
            
            # 保存mask（如果存在）
            if result.mask_image:
                mask_path = output_path.parent / f"{output_path.stem}_mask{output_path.suffix}"
                result.mask_image.save(mask_path)
                logger.info(f"✅ Mask saved: {mask_path}")
        else:
            logger.error(f"❌ Processing failed: {result.error_message}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Error processing file: {e}")
        sys.exit(1)

def process_directory(args, processor):
    """批量处理目录"""
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    
    # 获取所有图像文件
    image_files = [f for f in input_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        logger.error(f"No image files found in: {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(image_files)} image files")
    
    # 批量处理
    success_count = 0
    for i, image_file in enumerate(image_files, 1):
        try:
            logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            # 输出文件路径
            output_file = output_dir / f"processed_{image_file.name}"
            
            # 处理单个文件
            result = processor.process_image(
                image=Image.open(image_file),
                mask_method=args.mask_method if args.mask_method != "auto" else "custom",
                transparent=args.transparent,
                max_bbox_percent=args.max_bbox_percent,
                force_format=args.force_format
            )
            
            if result.success:
                result.result_image.save(output_file)
                success_count += 1
                logger.info(f"✅ {image_file.name} -> {output_file.name}")
            else:
                logger.error(f"❌ Failed to process {image_file.name}: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {image_file.name}: {e}")
    
    logger.info(f"Batch processing completed: {success_count}/{len(image_files)} successful") 