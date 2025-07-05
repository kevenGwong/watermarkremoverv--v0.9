import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import tempfile
from PIL import Image
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from interfaces.web.ui import MainInterface, ParameterPanel
from config.config import ConfigManager
from core.processors.processing_result import ProcessingResult


class TestParameterPanel(unittest.TestCase):
    """测试参数面板组件"""
    
    def setUp(self):
        """设置测试环境"""
        self.config_manager = Mock(spec=ConfigManager)
        self.parameter_panel = ParameterPanel(self.config_manager)
    
    def test_render_returns_correct_tuple(self):
        """测试render方法返回正确的元组格式"""
        with patch('streamlit.columns'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.slider'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.text_input'):
            
            result = self.parameter_panel.render()
            
            # 检查返回类型
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 5)
            
            # 检查各个元素的类型
            mask_model, mask_params, inpaint_params, performance_params, transparent = result
            
            self.assertIsInstance(mask_model, str)
            self.assertIsInstance(mask_params, dict)
            self.assertIsInstance(inpaint_params, dict)
            self.assertIsInstance(performance_params, dict)
            self.assertIsInstance(transparent, bool)
    
    def test_mask_params_structure(self):
        """测试mask参数的结构"""
        with patch('streamlit.columns'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.slider'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.text_input'):
            
            _, mask_params, _, _, _ = self.parameter_panel.render()
            
            # 检查必要的参数是否存在
            self.assertIn('mask_threshold', mask_params)
            self.assertIn('mask_dilate_kernel_size', mask_params)
            self.assertIn('mask_dilate_iterations', mask_params)
            
            # 检查参数值的合理性
            self.assertGreaterEqual(mask_params['mask_threshold'], 0.0)
            self.assertLessEqual(mask_params['mask_threshold'], 1.0)
            self.assertGreaterEqual(mask_params['mask_dilate_kernel_size'], 1)
            self.assertGreaterEqual(mask_params['mask_dilate_iterations'], 0)
    
    def test_inpaint_params_structure(self):
        """测试inpaint参数的结构"""
        with patch('streamlit.columns'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.slider'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.text_input'):
            
            _, _, inpaint_params, _, _ = self.parameter_panel.render()
            
            # 检查必要的参数是否存在
            self.assertIn('inpaint_model', inpaint_params)
            
            # 检查模型选择的有效性
            valid_models = ['lama', 'iopaint']
            self.assertIn(inpaint_params['inpaint_model'], valid_models)


class TestMainInterface(unittest.TestCase):
    """测试主界面组件"""
    
    def setUp(self):
        """设置测试环境"""
        self.config_manager = Mock(spec=ConfigManager)
        self.main_interface = MainInterface(self.config_manager)
        
        # 创建测试图片
        self.test_image = Image.new('RGB', (100, 100), color='red')
    
    def test_check_parameter_changes(self):
        """测试参数变化检测功能"""
        # 模拟session_state
        with patch('streamlit.session_state', {}) as mock_session:
            # 第一次调用，应该没有变化
            self.main_interface._check_parameter_changes(
                'test_model', 
                {'param1': 1}, 
                {'param2': 2}, 
                {'param3': 3}, 
                False
            )
            
            # 检查是否保存了当前参数
            self.assertIn('current_parameters', mock_session)
            
            # 模拟有之前的参数
            mock_session['last_parameters'] = {
                'mask_model': 'old_model',
                'mask_params': {'param1': 0},
                'inpaint_params': {'param2': 1},
                'performance_params': {'param3': 2},
                'transparent': True
            }
            
            # 再次调用，应该检测到变化
            self.main_interface._check_parameter_changes(
                'test_model', 
                {'param1': 1}, 
                {'param2': 2}, 
                {'param3': 3}, 
                False
            )
    
    def test_render_parameter_summary(self):
        """测试参数总结渲染"""
        with patch('streamlit.expander'), \
             patch('streamlit.columns'), \
             patch('streamlit.write'):
            
            # 测试正常情况
            self.main_interface._render_parameter_summary(
                'test_model',
                {'param1': 1},
                {'inpaint_model': 'lama', 'param2': 2},
                {'param3': 3},
                False
            )
            
            # 测试IOPaint模型
            self.main_interface._render_parameter_summary(
                'test_model',
                {'param1': 1},
                {'inpaint_model': 'iopaint', 'force_model': 'lama', 'param2': 2},
                {'param3': 3},
                True
            )
    
    def test_process_button_rendering(self):
        """测试处理按钮渲染"""
        mock_inference_manager = Mock()
        
        with patch('streamlit.columns'), \
             patch('streamlit.button'), \
             patch('streamlit.spinner'), \
             patch('streamlit.error'):
            
            self.main_interface._render_process_button(
                mock_inference_manager,
                self.test_image,
                'test_model',
                {'param1': 1},
                {'param2': 2},
                {'param3': 3},
                False
            )
    
    def test_results_rendering(self):
        """测试结果渲染"""
        # 创建测试结果
        test_result = ProcessingResult(
            success=True,
            result_image=self.test_image,
            mask_image=Image.new('L', (100, 100), 128),
            processing_time=1.5,
            error_message=None
        )
        
        with patch('streamlit.subheader'), \
             patch('streamlit.columns'), \
             patch('streamlit.metric'), \
             patch('streamlit.warning'):
            
            self.main_interface._render_results(
                test_result,
                self.test_image,
                False,
                'test.jpg'
            )


class TestUIErrorHandling(unittest.TestCase):
    """测试UI错误处理"""
    
    def setUp(self):
        """设置测试环境"""
        self.config_manager = Mock(spec=ConfigManager)
        self.main_interface = MainInterface(self.config_manager)
    
    def test_invalid_parameter_values(self):
        """测试无效参数值的处理"""
        with patch('streamlit.columns'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.slider'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.text_input'):
            
            # 测试边界值
            _, mask_params, _, _, _ = self.parameter_panel.render()
            
            # 确保参数在合理范围内
            self.assertGreaterEqual(mask_params['mask_threshold'], 0.0)
            self.assertLessEqual(mask_params['mask_threshold'], 1.0)
    
    def test_missing_parameters(self):
        """测试缺失参数的处理"""
        # 测试参数面板是否能处理缺失的配置
        with patch('streamlit.columns'), \
             patch('streamlit.selectbox'), \
             patch('streamlit.slider'), \
             patch('streamlit.checkbox'), \
             patch('streamlit.text_input'):
            
            result = self.parameter_panel.render()
            self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main() 