#!/usr/bin/env python3
"""
测试IOPaint的实际颜色行为
确定IOPaint期望的输入输出格式
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_iopaint_color_expectation():
    """测试IOPaint对颜色格式的实际期望"""
    print("🧪 测试IOPaint颜色格式期望")
    print("=" * 50)
    
    try:
        # 导入IOPaint
        from iopaint.model_manager import ModelManager
        from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"设备: {device}")
        
        # 创建测试数据
        # 创建一个简单的红色图像
        test_image_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        test_image_rgb[:, :, 0] = 255  # 红色通道
        test_image_rgb[:, :, 1] = 50   # 绿色通道  
        test_image_rgb[:, :, 2] = 50   # 蓝色通道
        
        # 创建BGR版本用于对比
        test_image_bgr = test_image_rgb[:, :, ::-1].copy()  # BGR = RGB逆序
        
        # 创建mask（中心白色区域）
        test_mask = np.zeros((64, 64), dtype=np.uint8)
        test_mask[20:44, 20:44] = 255
        
        print(f"RGB测试图像 左上角像素: {test_image_rgb[0,0]}")
        print(f"BGR测试图像 左上角像素: {test_image_bgr[0,0]}")
        
        # 测试配置
        config = Config(
            ldm_steps=5,  # 最少步数
            ldm_sampler=LDMSampler.ddim,
            hd_strategy=HDStrategy.ORIGINAL
        )
        
        # 测试不同模型的颜色行为
        models_to_test = ['lama', 'mat']
        
        for model_name in models_to_test:
            print(f"\n🎨 测试 {model_name.upper()} 模型")
            print("-" * 30)
            
            try:
                # 加载模型
                model_manager = ModelManager(name=model_name, device=str(device))
                print(f"✅ {model_name.upper()} 模型加载成功")
                
                # 测试1: RGB输入
                print("   📥 测试RGB输入...")
                result_from_rgb = model_manager(test_image_rgb, test_mask, config)
                
                # 检查未被mask覆盖的区域（应该保持原样）
                unmask_rgb_result = result_from_rgb[0, 0]  # 左上角，未被mask覆盖
                print(f"   RGB输入 -> 未处理区域输出: {unmask_rgb_result}")
                
                # 测试2: BGR输入
                print("   📥 测试BGR输入...")
                result_from_bgr = model_manager(test_image_bgr, test_mask, config)
                
                unmask_bgr_result = result_from_bgr[0, 0]
                print(f"   BGR输入 -> 未处理区域输出: {unmask_bgr_result}")
                
                # 分析结果
                print("\n   📊 分析结果:")
                print(f"   原始RGB: {test_image_rgb[0,0]}")
                print(f"   原始BGR: {test_image_bgr[0,0]}")
                print(f"   RGB输入结果: {unmask_rgb_result}")
                print(f"   BGR输入结果: {unmask_bgr_result}")
                
                # 判断IOPaint期望的格式
                rgb_preserved = np.allclose(unmask_rgb_result, test_image_rgb[0,0], atol=5)
                bgr_preserved = np.allclose(unmask_bgr_result, test_image_bgr[0,0], atol=5)
                
                rgb_to_bgr = np.allclose(unmask_rgb_result, test_image_bgr[0,0], atol=5)
                bgr_to_rgb = np.allclose(unmask_bgr_result, test_image_rgb[0,0], atol=5)
                
                print(f"\n   🔍 匹配分析:")
                print(f"   RGB输入保持RGB: {rgb_preserved}")
                print(f"   BGR输入保持BGR: {bgr_preserved}")  
                print(f"   RGB输入变BGR: {rgb_to_bgr}")
                print(f"   BGR输入变RGB: {bgr_to_rgb}")
                
                if rgb_preserved:
                    print(f"   ✅ {model_name.upper()} 期望RGB输入，输出RGB")
                elif rgb_to_bgr:
                    print(f"   ⚠️ {model_name.upper()} 期望RGB输入，但输出BGR")
                elif bgr_preserved:
                    print(f"   ⚠️ {model_name.upper()} 期望BGR输入，输出BGR")
                elif bgr_to_rgb:
                    print(f"   ⚠️ {model_name.upper()} 期望BGR输入，但输出RGB")
                else:
                    print(f"   ❌ {model_name.upper()} 颜色行为无法确定")
                
                # 清理
                del model_manager
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"   ❌ {model_name.upper()} 测试失败: {e}")
                continue
        
    except Exception as e:
        print(f"❌ IOPaint测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_opencv_in_iopaint():
    """检查IOPaint内部是否使用了OpenCV"""
    print("\n🔍 检查IOPaint内部OpenCV使用")
    print("=" * 40)
    
    try:
        import iopaint
        import inspect
        import os
        
        # 获取iopaint模块路径
        iopaint_path = os.path.dirname(iopaint.__file__)
        print(f"IOPaint路径: {iopaint_path}")
        
        # 搜索关键文件中的颜色转换代码
        key_files = [
            'model_manager.py',
            'helper.py',
            'model/__init__.py'
        ]
        
        for file_name in key_files:
            file_path = os.path.join(iopaint_path, file_name)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    if 'cv2.cvtColor' in content or 'COLOR_RGB2BGR' in content or 'COLOR_BGR2RGB' in content:
                        print(f"⚠️ {file_name} 中发现颜色转换代码")
                        
                        # 提取相关行
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'cv2.cvtColor' in line or 'COLOR_' in line:
                                print(f"   第{i+1}行: {line.strip()}")
                    else:
                        print(f"✅ {file_name} 中未发现颜色转换")
                        
                except Exception as e:
                    print(f"⚠️ 无法读取 {file_name}: {e}")
            else:
                print(f"⚠️ {file_name} 不存在")
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")

def main():
    """主函数"""
    print("🔬 IOPaint颜色行为深度分析")
    print("=" * 60)
    
    # 测试IOPaint的颜色期望
    test_iopaint_color_expectation()
    
    # 检查IOPaint内部的OpenCV使用
    test_opencv_in_iopaint()
    
    print("\n" + "=" * 60)
    print("🎯 建议根据上述结果调整颜色处理策略")

if __name__ == "__main__":
    main()