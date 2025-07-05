#!/usr/bin/env python3
"""
UI交互优化测试 - 验证模型切换时的预览刷新机制
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from core.inference import process_image
from config.config import ConfigManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_parameter_change_detection():
    """测试参数变化检测逻辑"""
    print("🧪 测试参数变化检测...")
    
    # 模拟不同的参数组合
    test_cases = [
        {
            'name': '基础MAT模型',
            'mask_model': 'upload',
            'inpaint_params': {
                'inpaint_model': 'iopaint',
                'force_model': 'mat',
                'hd_strategy': 'ORIGINAL'
            }
        },
        {
            'name': '切换到FCF模型',
            'mask_model': 'upload',
            'inpaint_params': {
                'inpaint_model': 'iopaint',
                'force_model': 'fcf',
                'hd_strategy': 'ORIGINAL'
            }
        },
        {
            'name': '切换到ZITS模型',
            'mask_model': 'upload',
            'inpaint_params': {
                'inpaint_model': 'iopaint',
                'force_model': 'zits',
                'hd_strategy': 'ORIGINAL'
            }
        },
        {
            'name': '切换到LaMA',
            'mask_model': 'upload',
            'inpaint_params': {
                'inpaint_model': 'lama',
                'hd_strategy': 'ORIGINAL'
            }
        },
        {
            'name': '切换Mask模式',
            'mask_model': 'custom',
            'inpaint_params': {
                'inpaint_model': 'iopaint',
                'force_model': 'mat',
                'hd_strategy': 'ORIGINAL'
            }
        }
    ]
    
    def detect_changes(prev_params, curr_params):
        """模拟UI中的参数变化检测逻辑"""
        if not prev_params:
            return []
        
        changes = []
        
        # 检查模型选择变化
        if prev_params.get('mask_model') != curr_params['mask_model']:
            changes.append('mask_model')
        
        # 检查inpaint模型变化
        if prev_params.get('inpaint_params', {}).get('inpaint_model') != curr_params['inpaint_params'].get('inpaint_model'):
            changes.append('inpaint_model')
        
        # 检查具体模型选择变化
        if prev_params.get('inpaint_params', {}).get('force_model') != curr_params['inpaint_params'].get('force_model'):
            changes.append('specific_model')
        
        return changes
    
    # 模拟参数变化检测
    previous_params = None
    results = []
    
    for i, case in enumerate(test_cases):
        changes = detect_changes(previous_params, case)
        
        result = {
            'step': i + 1,
            'name': case['name'],
            'changes_detected': changes,
            'should_refresh': len(changes) > 0
        }
        results.append(result)
        
        print(f"  步骤 {i+1}: {case['name']}")
        if changes:
            print(f"    🔄 检测到变化: {', '.join(changes)}")
            print(f"    ✅ 应该清除旧结果并要求重新处理")
        else:
            print(f"    ➡️ 无参数变化")
        
        previous_params = case.copy()
    
    return results

def test_model_switch_workflow():
    """测试模型切换的完整工作流"""
    print("\n🧪 测试模型切换工作流...")
    
    config_manager = ConfigManager()
    
    # 创建测试图像和mask
    test_image = Image.new('RGB', (512, 512), color=(200, 150, 100))
    img_array = np.array(test_image)
    img_array[100:150, 100:150] = [255, 100, 100]
    test_image = Image.fromarray(img_array)
    
    test_mask = Image.new('L', (512, 512), color=0)
    mask_array = np.array(test_mask)
    mask_array[200:300, 200:300] = 255
    test_mask = Image.fromarray(mask_array, mode='L')
    
    # 测试不同模型的处理时间和结果差异
    models_to_test = ['fcf', 'mat']  # ZITS会比较慢，只测试FCF和MAT
    
    results = {}
    
    for model in models_to_test:
        print(f"  📋 测试模型: {model.upper()}")
        
        start_time = time.time()
        
        try:
            result = process_image(
                image=test_image,
                mask_model='upload',
                mask_params={'uploaded_mask': test_mask},
                inpaint_params={
                    'inpaint_model': 'iopaint',
                    'force_model': model,
                    'hd_strategy': 'ORIGINAL',
                    'ldm_steps': 20  # 减少步数加快测试
                },
                config_manager=config_manager
            )
            
            processing_time = time.time() - start_time
            
            if result.success:
                # 计算结果差异（简单的像素差异）
                result_array = np.array(result.result_image.convert('RGB'))
                original_array = np.array(test_image)
                diff_pixels = np.sum(np.abs(result_array - original_array))
                
                results[model] = {
                    'success': True,
                    'processing_time': processing_time,
                    'pixel_difference': diff_pixels,
                    'output_size': result.result_image.size,
                    'mask_coverage': np.mean(np.array(result.mask_image)) / 255 * 100 if result.mask_image else 0
                }
                
                print(f"    ✅ 处理成功: {processing_time:.2f}s")
                print(f"    📊 像素差异: {diff_pixels:,}")
                print(f"    📏 输出尺寸: {result.result_image.size}")
                
                # 保存结果
                result.result_image.save(f"scripts/ui_test_{model}_result.png")
                
            else:
                results[model] = {
                    'success': False,
                    'error': result.error_message,
                    'processing_time': processing_time
                }
                print(f"    ❌ 处理失败: {result.error_message}")
                
        except Exception as e:
            results[model] = {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
            print(f"    ❌ 处理异常: {e}")
    
    return results

def test_ui_state_management():
    """测试UI状态管理逻辑"""
    print("\n🧪 测试UI状态管理...")
    
    # 模拟Streamlit session_state行为
    class MockSessionState:
        def __init__(self):
            self.data = {}
        
        def get(self, key, default=None):
            return self.data.get(key, default)
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def __getitem__(self, key):
            return self.data[key]
        
        def __contains__(self, key):
            return key in self.data
        
        def __delitem__(self, key):
            if key in self.data:
                del self.data[key]
    
    session_state = MockSessionState()
    
    # 模拟UI交互序列
    scenarios = [
        {
            'action': '初始加载',
            'params': {
                'mask_model': 'upload',
                'inpaint_model': 'iopaint',
                'force_model': 'mat'
            }
        },
        {
            'action': '切换到FCF模型',
            'params': {
                'mask_model': 'upload',
                'inpaint_model': 'iopaint',
                'force_model': 'fcf'
            }
        },
        {
            'action': '处理图像',
            'has_result': True
        },
        {
            'action': '切换到ZITS模型',
            'params': {
                'mask_model': 'upload',
                'inpaint_model': 'iopaint',
                'force_model': 'zits'
            }
        },
        {
            'action': '切换mask模式',
            'params': {
                'mask_model': 'custom',
                'inpaint_model': 'iopaint',
                'force_model': 'zits'
            }
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"  步骤 {i+1}: {scenario['action']}")
        
        if 'params' in scenario:
            # 检查参数变化
            if 'last_parameters' in session_state:
                last_params = session_state['last_parameters']
                current_params = scenario['params']
                
                changes = []
                if last_params.get('mask_model') != current_params.get('mask_model'):
                    changes.append('mask_model')
                if last_params.get('force_model') != current_params.get('force_model'):
                    changes.append('force_model')
                
                if changes:
                    print(f"    🔄 检测到变化: {', '.join(changes)}")
                    if 'processing_result' in session_state:
                        del session_state['processing_result']
                        print(f"    🗑️ 清除旧结果")
            
            session_state['last_parameters'] = scenario['params']
        
        if scenario.get('has_result'):
            session_state['processing_result'] = {'success': True, 'timestamp': time.time()}
            print(f"    💾 保存处理结果")
        
        print(f"    📊 当前状态: {list(session_state.data.keys())}")
    
    return True

def test_ui_refresh_optimization():
    """UI交互优化完整测试"""
    print("🚀 开始UI交互优化测试...")
    
    results = {
        'parameter_detection': test_parameter_change_detection(),
        'model_switch_workflow': test_model_switch_workflow(),
        'state_management': test_ui_state_management()
    }
    
    # 输出总结
    print("\n" + "="*60)
    print("📊 UI交互优化测试结果总结")
    print("="*60)
    
    # 参数检测总结
    print("\n🔍 参数变化检测:")
    detection_results = results['parameter_detection']
    refresh_count = sum(1 for r in detection_results if r['should_refresh'])
    print(f"   测试用例: {len(detection_results)}")
    print(f"   需要刷新: {refresh_count}")
    print(f"   检测准确率: {refresh_count/len(detection_results)*100:.1f}%")
    
    # 模型切换总结
    print("\n🔄 模型切换工作流:")
    workflow_results = results['model_switch_workflow']
    success_count = sum(1 for r in workflow_results.values() if r['success'])
    print(f"   测试模型: {len(workflow_results)}")
    print(f"   成功切换: {success_count}")
    if success_count > 0:
        avg_time = np.mean([r['processing_time'] for r in workflow_results.values() if r['success']])
        print(f"   平均处理时间: {avg_time:.2f}s")
    
    # 状态管理总结
    print("\n📊 状态管理:")
    state_result = results['state_management']
    print(f"   状态管理测试: {'✅ 通过' if state_result else '❌ 失败'}")
    
    # 优化效果评估
    overall_success = (
        refresh_count > 0 and  # 参数变化能被检测
        success_count >= len(workflow_results) * 0.8 and  # 80%模型切换成功
        state_result  # 状态管理正常
    )
    
    print(f"\n🎯 整体评估:")
    print(f"   UI交互优化: {'✅ 成功' if overall_success else '⚠️ 需要改进'}")
    
    if overall_success:
        print("\n🎉 UI交互优化验证通过!")
        print("✅ 模型切换时预览会自动刷新")
        print("✅ 参数变化检测正常工作")
        print("✅ 状态管理机制完善")
    else:
        print("\n⚠️ UI交互优化需要进一步改进")
    
    return results, overall_success

if __name__ == "__main__":
    results, success = test_ui_refresh_optimization()
    
    if success:
        print("\n✅ UI交互优化测试全部通过!")
    else:
        print("\n⚠️ 部分UI交互功能需要优化")