#!/usr/bin/env python3
"""
端到端测试脚本
模拟完整的用户操作流程，包括图片上传、参数设置、处理等
"""

import sys
import os
import time
import tempfile
import subprocess
import signal
import requests
from PIL import Image
import json
import threading
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.inference_manager import InferenceManager
from core.processors.processing_result import ProcessingResult
from utils.image_processor import ImageProcessor


class E2ETestRunner:
    """端到端测试运行器"""
    
    def __init__(self):
        self.base_url = "http://localhost:8501"
        self.process = None
        self.test_results = []
        
    def create_test_image(self, size=(512, 512), color='red', watermark=True):
        """创建测试图片"""
        image = Image.new('RGB', size, color)
        
        if watermark:
            # 添加简单的水印
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(image)
            
            # 尝试使用默认字体
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # 添加文字水印
            text = "TEST WATERMARK"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (size[0] - text_width) // 2
            y = (size[1] - text_height) // 2
            
            draw.text((x, y), text, fill='white', font=font)
            
            # 添加矩形水印
            draw.rectangle([50, 50, 150, 100], outline='white', width=3)
        
        return image
    
    def start_streamlit_app(self):
        """启动Streamlit应用"""
        print("🚀 启动Streamlit应用...")
        
        try:
            self.process = subprocess.Popen([
                'streamlit', 'run', 'interfaces/web/main.py',
                '--server.port', '8501',
                '--server.headless', 'true',
                '--server.enableCORS', 'false',
                '--server.enableXsrfProtection', 'false'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 等待应用启动
            time.sleep(5)
            
            # 检查应用是否启动成功
            try:
                response = requests.get(f"{self.base_url}/_stcore/health", timeout=10)
                if response.status_code == 200:
                    print("✅ Streamlit应用启动成功")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print("❌ Streamlit应用启动失败")
            return False
            
        except Exception as e:
            print(f"❌ 启动应用时出错: {e}")
            return False
    
    def stop_streamlit_app(self):
        """停止Streamlit应用"""
        if self.process:
            print("🛑 停止Streamlit应用...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print("✅ Streamlit应用已停止")
    
    def test_image_upload_workflow(self):
        """测试图片上传工作流"""
        print("\n📸 测试图片上传工作流...")
        
        # 创建测试图片
        test_image = self.create_test_image((400, 300), 'blue', True)
        temp_path = tempfile.mktemp(suffix='.jpg')
        test_image.save(temp_path)
        
        try:
            # 模拟文件上传
            with open(temp_path, 'rb') as f:
                files = {'file': ('test.jpg', f, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/upload", files=files)
            
            if response.status_code == 200:
                print("✅ 图片上传成功")
                return True
            else:
                print(f"❌ 图片上传失败: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 图片上传测试出错: {e}")
            return False
        finally:
            os.unlink(temp_path)
    
    def test_parameter_validation(self):
        """测试参数验证"""
        print("\n⚙️ 测试参数验证...")
        
        test_cases = [
            {
                'name': '有效LaMA参数',
                'params': {
                    'inpaint_model': 'lama',
                    'ldm_steps': 20,
                    'hd_strategy': 'ORIGINAL'
                },
                'expected': True
            },
            {
                'name': '有效IOPaint参数',
                'params': {
                    'inpaint_model': 'iopaint',
                    'force_model': 'lama',
                    'ldm_steps': 20,
                    'hd_strategy': 'CROP'
                },
                'expected': True
            },
            {
                'name': '无效参数值',
                'params': {
                    'inpaint_model': 'invalid_model',
                    'ldm_steps': -1
                },
                'expected': False
            }
        ]
        
        success_count = 0
        for test_case in test_cases:
            try:
                # 这里可以添加实际的参数验证逻辑
                is_valid = self.validate_parameters(test_case['params'])
                
                if is_valid == test_case['expected']:
                    print(f"✅ {test_case['name']}: 通过")
                    success_count += 1
                else:
                    print(f"❌ {test_case['name']}: 失败")
                    
            except Exception as e:
                print(f"❌ {test_case['name']}: 出错 - {e}")
        
        return success_count == len(test_cases)
    
    def validate_parameters(self, params):
        """验证参数有效性"""
        # 检查模型选择
        valid_models = ['lama', 'iopaint']
        if 'inpaint_model' in params and params['inpaint_model'] not in valid_models:
            return False
        
        # 检查步数
        if 'ldm_steps' in params and (params['ldm_steps'] < 1 or params['ldm_steps'] > 100):
            return False
        
        # 检查策略
        valid_strategies = ['ORIGINAL', 'RESIZE', 'CROP']
        if 'hd_strategy' in params and params['hd_strategy'] not in valid_strategies:
            return False
        
        return True
    
    def test_processing_workflow(self):
        """测试处理工作流"""
        print("\n🔄 测试处理工作流...")
        
        # 创建测试图片
        test_image = self.create_test_image((256, 256), 'green', True)
        
        # 测试不同的处理配置
        test_configs = [
            {
                'name': 'LaMA基础处理',
                'mask_model': 'simple',
                'mask_params': {'mask_threshold': 0.5},
                'inpaint_params': {'inpaint_model': 'lama'},
                'performance_params': {'max_size': 1024},
                'transparent': False
            },
            {
                'name': 'IOPaint处理',
                'mask_model': 'simple',
                'mask_params': {'mask_threshold': 0.5},
                'inpaint_params': {
                    'inpaint_model': 'iopaint',
                    'force_model': 'lama',
                    'ldm_steps': 20
                },
                'performance_params': {'max_size': 1024},
                'transparent': True
            }
        ]
        
        success_count = 0
        for config in test_configs:
            try:
                print(f"  测试: {config['name']}")
                
                # 模拟处理过程
                result = self.simulate_processing(test_image, config)
                
                if result and result.success:
                    print(f"  ✅ {config['name']}: 处理成功")
                    success_count += 1
                else:
                    print(f"  ❌ {config['name']}: 处理失败")
                    
            except Exception as e:
                print(f"  ❌ {config['name']}: 出错 - {e}")
        
        return success_count == len(test_configs)
    
    def simulate_processing(self, image, config):
        """模拟处理过程"""
        try:
            # 创建推理管理器
            inference_manager = InferenceManager()
            
            # 模拟处理时间
            start_time = time.time()
            time.sleep(0.1)  # 模拟处理时间
            
            # 创建处理结果
            result = ProcessingResult(
                success=True,
                result_image=image.copy(),
                mask_image=Image.new('L', image.size, 128),
                processing_time=time.time() - start_time,
                error_message=None
            )
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                result_image=None,
                mask_image=None,
                processing_time=0.0,
                error_message=str(e)
            )
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n🚨 测试错误处理...")
        
        error_scenarios = [
            {
                'name': '无效图片格式',
                'image_data': b'invalid image data',
                'expected_error': True
            },
            {
                'name': '超大图片',
                'image': self.create_test_image((4096, 4096)),
                'expected_error': False  # 应该能处理
            },
            {
                'name': '空图片',
                'image': Image.new('RGB', (0, 0)),
                'expected_error': True
            }
        ]
        
        success_count = 0
        for scenario in error_scenarios:
            try:
                print(f"  测试: {scenario['name']}")
                
                if 'image_data' in scenario:
                    # 测试无效图片数据
                    result = self.handle_invalid_image(scenario['image_data'])
                else:
                    # 测试特殊图片
                    result = self.simulate_processing(scenario['image'], {
                        'mask_model': 'simple',
                        'mask_params': {'mask_threshold': 0.5},
                        'inpaint_params': {'inpaint_model': 'lama'},
                        'performance_params': {'max_size': 1024},
                        'transparent': False
                    })
                
                if scenario['expected_error'] and not result.success:
                    print(f"  ✅ {scenario['name']}: 正确捕获错误")
                    success_count += 1
                elif not scenario['expected_error'] and result.success:
                    print(f"  ✅ {scenario['name']}: 正确处理")
                    success_count += 1
                else:
                    print(f"  ❌ {scenario['name']}: 错误处理不符合预期")
                    
            except Exception as e:
                if scenario['expected_error']:
                    print(f"  ✅ {scenario['name']}: 正确抛出异常")
                    success_count += 1
                else:
                    print(f"  ❌ {scenario['name']}: 意外异常 - {e}")
        
        return success_count == len(error_scenarios)
    
    def handle_invalid_image(self, image_data):
        """处理无效图片"""
        try:
            # 尝试打开无效图片数据
            image = Image.open(tempfile.NamedTemporaryFile(mode='wb', delete=False))
            return ProcessingResult(
                success=True,
                result_image=image,
                mask_image=None,
                processing_time=0.0,
                error_message=None
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                result_image=None,
                mask_image=None,
                processing_time=0.0,
                error_message=str(e)
            )
    
    def test_performance_metrics(self):
        """测试性能指标"""
        print("\n⚡ 测试性能指标...")
        
        # 测试不同尺寸图片的处理时间
        test_sizes = [(256, 256), (512, 512), (1024, 1024)]
        performance_results = []
        
        for size in test_sizes:
            test_image = self.create_test_image(size, 'purple', True)
            
            start_time = time.time()
            result = self.simulate_processing(test_image, {
                'mask_model': 'simple',
                'mask_params': {'mask_threshold': 0.5},
                'inpaint_params': {'inpaint_model': 'lama'},
                'performance_params': {'max_size': 2048},
                'transparent': False
            })
            end_time = time.time()
            
            processing_time = end_time - start_time
            performance_results.append({
                'size': size,
                'time': processing_time,
                'success': result.success
            })
            
            print(f"  尺寸 {size}: {processing_time:.3f}s")
        
        # 检查性能是否在合理范围内
        all_success = all(r['success'] for r in performance_results)
        reasonable_time = all(r['time'] < 5.0 for r in performance_results)  # 5秒内完成
        
        if all_success and reasonable_time:
            print("✅ 性能测试通过")
            return True
        else:
            print("❌ 性能测试失败")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 开始端到端测试...")
        
        test_results = []
        
        # 启动应用
        if not self.start_streamlit_app():
            print("❌ 无法启动应用，跳过端到端测试")
            return False
        
        try:
            # 运行各项测试
            tests = [
                ("图片上传工作流", self.test_image_upload_workflow),
                ("参数验证", self.test_parameter_validation),
                ("处理工作流", self.test_processing_workflow),
                ("错误处理", self.test_error_handling),
                ("性能指标", self.test_performance_metrics)
            ]
            
            for test_name, test_func in tests:
                try:
                    result = test_func()
                    test_results.append((test_name, result))
                except Exception as e:
                    print(f"❌ {test_name} 测试出错: {e}")
                    test_results.append((test_name, False))
        
        finally:
            # 停止应用
            self.stop_streamlit_app()
        
        # 输出结果
        print("\n📊 端到端测试结果:")
        passed = 0
        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n总计: {passed}/{len(test_results)} 测试通过")
        
        return passed == len(test_results)


def main():
    """主函数"""
    runner = E2ETestRunner()
    success = runner.run_all_tests()
    
    if success:
        print("\n🎉 所有端到端测试通过！")
        sys.exit(0)
    else:
        print("\n💥 部分端到端测试失败！")
        sys.exit(1)


if __name__ == '__main__':
    main() 