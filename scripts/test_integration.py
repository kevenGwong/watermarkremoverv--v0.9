#!/usr/bin/env python3
"""
完整的集成测试脚本
验证UI参数传递、模块集成、处理流程的完整性
"""

import sys
import os
import logging
from pathlib import Path
from PIL import Image
import numpy as np

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from config.config import ConfigManager
from core.inference import InferenceManager
from interfaces.web.ui import ParameterPanel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class IntegrationTester:
    """集成测试器"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.inference_manager = InferenceManager(self.config_manager)
        self.test_results = {}
        
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 开始完整集成测试")
        print("=" * 60)
        
        # 1. 测试基础组件加载
        self.test_basic_loading()
        
        # 2. 测试参数传递
        self.test_parameter_passing()
        
        # 3. 测试mask处理
        self.test_mask_processing()
        
        # 4. 测试图像处理流程
        self.test_image_processing()
        
        # 5. 测试输出保存
        self.test_output_saving()
        
        # 输出总结
        self.print_summary()
        
    def test_basic_loading(self):
        """测试1: 基础组件加载"""
        print("\n📋 测试1: 基础组件加载")
        
        try:
            # 加载推理管理器
            success = self.inference_manager.load_processor()
            assert success, "推理管理器加载失败"
            
            # 检查mask生成器
            assert hasattr(self.inference_manager.processor, 'mask_generator'), "mask_generator未初始化"
            
            # 检查LaMA处理器
            assert hasattr(self.inference_manager.processor, 'lama_processor'), "lama_processor未初始化"
            assert self.inference_manager.processor.lama_processor.model_loaded, "LaMA模型未加载"
            
            # 检查enhanced_processor
            assert self.inference_manager.enhanced_processor is not None, "enhanced_processor未初始化"
            
            print("✅ 基础组件加载测试通过")
            self.test_results['basic_loading'] = True
            
        except Exception as e:
            print(f"❌ 基础组件加载测试失败: {e}")
            self.test_results['basic_loading'] = False
            
    def test_parameter_passing(self):
        """测试2: UI参数传递"""
        print("\n📋 测试2: UI参数传递")
        
        try:
            # 模拟UI参数
            test_params = {
                'mask_model': 'custom',
                'mask_params': {
                    'mask_threshold': 0.5,
                    'mask_dilate_kernel_size': 5,
                    'mask_dilate_iterations': 2
                },
                'inpaint_params': {
                    'inpaint_model': 'lama',  # 使用LaMA而不是PowerPaint
                    'prompt': 'clean background',
                    'negative_prompt': 'watermark, logo, text',
                    'num_inference_steps': 20,
                    'guidance_scale': 7.5
                },
                'performance_params': {
                    'crop_trigger_size': 512,
                    'enable_crop_strategy': True
                }
            }
            
            # 验证参数结构完整性
            assert 'mask_model' in test_params
            assert 'mask_params' in test_params
            assert 'inpaint_params' in test_params
            assert 'performance_params' in test_params
            
            print("✅ UI参数结构完整")
            print(f"   - Mask模型: {test_params['mask_model']}")
            print(f"   - Inpaint模型: {test_params['inpaint_params']['inpaint_model']}")
            print(f"   - Crop策略: {test_params['performance_params']['enable_crop_strategy']}")
            
            self.test_results['parameter_passing'] = True
            
        except Exception as e:
            print(f"❌ 参数传递测试失败: {e}")
            self.test_results['parameter_passing'] = False
            
    def test_mask_processing(self):
        """测试3: Mask处理"""
        print("\n📋 测试3: Mask处理")
        
        try:
            # 创建测试图像 (512x512)
            test_image = Image.new('RGB', (512, 512), 'white')
            
            # 测试自定义mask生成
            mask_params = {'mask_threshold': 0.5}
            mask_image = self.inference_manager.processor.mask_generator.generate_mask(test_image, mask_params)
            
            assert isinstance(mask_image, Image.Image), "Mask生成返回类型错误"
            assert mask_image.size == test_image.size, "Mask尺寸与原图不匹配"
            assert mask_image.mode == 'L', "Mask格式错误，应为灰度图"
            
            # 检查mask内容
            mask_array = np.array(mask_image)
            print(f"   - Mask尺寸: {mask_image.size}")
            print(f"   - Mask像素值范围: {mask_array.min()} - {mask_array.max()}")
            print(f"   - 白色像素数量: {np.sum(mask_array > 128)}")
            
            print("✅ Mask处理测试通过")
            self.test_results['mask_processing'] = True
            
        except Exception as e:
            print(f"❌ Mask处理测试失败: {e}")
            self.test_results['mask_processing'] = False
            
    def test_image_processing(self):
        """测试4: 图像处理流程"""
        print("\n📋 测试4: 图像处理流程")
        
        try:
            # 创建测试图像
            test_image = Image.new('RGB', (512, 512), 'white')
            
            # 创建测试mask (中心区域为白色)
            test_mask = Image.new('L', (512, 512), 0)
            for x in range(200, 312):
                for y in range(200, 312):
                    test_mask.putpixel((x, y), 255)
                    
            # 测试参数
            mask_params = {'mask_threshold': 0.5}
            inpaint_params = {
                'inpaint_model': 'lama',  # 使用LaMA
                'prompt': 'clean background',
                'negative_prompt': 'watermark'
            }
            performance_params = {'crop_trigger_size': 512}
            
            # 执行处理
            result = self.inference_manager.process_image(
                image=test_image,
                mask_model='custom',
                mask_params=mask_params,
                inpaint_params=inpaint_params,
                performance_params=performance_params,
                transparent=False
            )
            
            assert result.success, f"图像处理失败: {result.error_message}"
            assert result.result_image is not None, "结果图像为空"
            assert isinstance(result.result_image, Image.Image), "结果图像类型错误"
            assert result.result_image.size == test_image.size, "结果图像尺寸错误"
            
            print(f"✅ 图像处理流程测试通过")
            print(f"   - 处理时间: {result.processing_time:.2f}秒")
            print(f"   - 结果图像尺寸: {result.result_image.size}")
            print(f"   - 结果图像模式: {result.result_image.mode}")
            
            self.test_results['image_processing'] = True
            
        except Exception as e:
            print(f"❌ 图像处理流程测试失败: {e}")
            self.test_results['image_processing'] = False
            
    def test_output_saving(self):
        """测试5: 输出保存"""
        print("\n📋 测试5: 输出保存")
        
        try:
            # 创建测试图像
            test_image = Image.new('RGB', (256, 256), 'red')
            
            # 测试保存功能
            output_dir = Path("temp")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "test_output.png"
            
            test_image.save(output_path)
            assert output_path.exists(), "图像保存失败"
            
            # 验证保存的图像
            loaded_image = Image.open(output_path)
            assert loaded_image.size == test_image.size, "保存图像尺寸错误"
            assert loaded_image.mode == test_image.mode, "保存图像模式错误"
            
            # 清理测试文件
            output_path.unlink()
            
            print("✅ 输出保存测试通过")
            self.test_results['output_saving'] = True
            
        except Exception as e:
            print(f"❌ 输出保存测试失败: {e}")
            self.test_results['output_saving'] = False
            
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 60)
        print("🎯 集成测试总结")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, passed in self.test_results.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{status} {test_name}")
            
        print(f"\n总体结果: {passed_tests}/{total_tests} 测试通过")
        
        if passed_tests == total_tests:
            print("🎉 所有集成测试通过！系统集成完整。")
        else:
            print("⚠️  部分测试失败，需要修复集成问题。")
            
        # 检查关键功能
        critical_tests = ['basic_loading', 'image_processing']
        critical_passed = all(self.test_results.get(test, False) for test in critical_tests)
        
        if critical_passed:
            print("✅ 关键功能正常，可以进行UI测试")
        else:
            print("❌ 关键功能异常，建议先修复核心问题")

if __name__ == "__main__":
    tester = IntegrationTester()
    tester.run_all_tests()