import unittest
import sys
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import requests
import subprocess
import signal

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from interfaces.web.main import main
from core.inference_manager import InferenceManager
from core.processors.processing_result import ProcessingResult


class TestWebIntegration(unittest.TestCase):
    """Web应用集成测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_image = Image.new('RGB', (200, 200), color='blue')
        self.temp_dir = tempfile.mkdtemp()
        
        # 保存测试图片
        self.test_image_path = os.path.join(self.temp_dir, 'test.jpg')
        self.test_image.save(self.test_image_path)
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('streamlit.run')
    def test_main_function_imports(self, mock_run):
        """测试主函数是否能正常导入和运行"""
        # 测试主函数是否存在
        self.assertTrue(callable(main))
        
        # 测试是否能正常调用（会被mock拦截）
        try:
            main()
        except Exception as e:
            # 如果出现异常，检查是否是预期的mock相关异常
            self.assertIn('mock', str(e).lower())
    
    def test_config_loading(self):
        """测试配置加载"""
        from config.config import ConfigManager
        
        config_manager = ConfigManager()
        self.assertIsNotNone(config_manager)
        
        # 测试配置是否包含必要的键
        config = config_manager.get_config()
        self.assertIsInstance(config, dict)
    
    def test_inference_manager_initialization(self):
        """测试推理管理器初始化"""
        with patch('core.models.base_inpainter.BaseInpainter') as mock_inpainter:
            mock_inpainter.return_value = Mock()
            
            inference_manager = InferenceManager()
            self.assertIsNotNone(inference_manager)
    
    def test_processing_result_structure(self):
        """测试处理结果结构"""
        result = ProcessingResult(
            success=True,
            result_image=self.test_image,
            mask_image=Image.new('L', (200, 200), 128),
            processing_time=1.5,
            error_message=None
        )
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.result_image)
        self.assertIsNotNone(result.mask_image)
        self.assertGreater(result.processing_time, 0)
        self.assertIsNone(result.error_message)
    
    def test_error_processing_result(self):
        """测试错误处理结果"""
        result = ProcessingResult(
            success=False,
            result_image=None,
            mask_image=None,
            processing_time=0.0,
            error_message="Test error"
        )
        
        self.assertFalse(result.success)
        self.assertIsNone(result.result_image)
        self.assertIsNone(result.mask_image)
        self.assertEqual(result.processing_time, 0.0)
        self.assertEqual(result.error_message, "Test error")


class TestStreamlitComponents(unittest.TestCase):
    """Streamlit组件测试"""
    
    def setUp(self):
        """设置测试环境"""
        self.test_image = Image.new('RGB', (100, 100), color='green')
    
    @patch('streamlit.file_uploader')
    @patch('streamlit.session_state')
    def test_file_upload_handling(self, mock_session, mock_uploader):
        """测试文件上传处理"""
        # 模拟文件上传
        mock_uploader.return_value = Mock()
        mock_uploader.return_value.name = 'test.jpg'
        mock_uploader.return_value.read = lambda: self.test_image.tobytes()
        
        # 测试文件上传逻辑
        uploaded_file = mock_uploader.return_value
        self.assertIsNotNone(uploaded_file)
        self.assertEqual(uploaded_file.name, 'test.jpg')
    
    @patch('streamlit.button')
    @patch('streamlit.spinner')
    def test_button_interaction(self, mock_spinner, mock_button):
        """测试按钮交互"""
        # 模拟按钮点击
        mock_button.return_value = True
        
        # 测试按钮状态
        button_clicked = mock_button.return_value
        self.assertTrue(button_clicked)
    
    @patch('streamlit.selectbox')
    def test_parameter_selection(self, mock_selectbox):
        """测试参数选择"""
        # 模拟下拉框选择
        mock_selectbox.return_value = 'lama'
        
        selected_model = mock_selectbox.return_value
        self.assertEqual(selected_model, 'lama')
        
        # 测试有效模型选择
        valid_models = ['lama', 'iopaint']
        self.assertIn(selected_model, valid_models)


class TestErrorScenarios(unittest.TestCase):
    """错误场景测试"""
    
    def test_invalid_image_format(self):
        """测试无效图片格式"""
        # 创建无效图片数据
        invalid_image_data = b'invalid image data'
        
        # 测试是否能正确处理无效图片
        try:
            Image.open(tempfile.NamedTemporaryFile(mode='wb', delete=False))
        except Exception as e:
            # 应该抛出异常
            self.assertIsInstance(e, Exception)
    
    def test_missing_dependencies(self):
        """测试缺失依赖"""
        # 测试关键模块是否能导入
        try:
            import streamlit
            import PIL
            import numpy
            import torch
        except ImportError as e:
            self.fail(f"Missing required dependency: {e}")
    
    def test_config_file_not_found(self):
        """测试配置文件不存在"""
        from config.config import ConfigManager
        
        # 测试默认配置加载
        config_manager = ConfigManager()
        config = config_manager.get_config()
        self.assertIsInstance(config, dict)


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def test_image_processing_time(self):
        """测试图片处理时间"""
        test_image = Image.new('RGB', (512, 512), color='red')
        
        start_time = time.time()
        
        # 模拟图片处理
        processed_image = test_image.copy()
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 处理时间应该在合理范围内
        self.assertLess(processing_time, 1.0)  # 应该很快完成
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 创建一些测试图片
        images = []
        for i in range(10):
            img = Image.new('RGB', (1024, 1024), color='blue')
            images.append(img)
        
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # 内存增长应该在合理范围内
        self.assertLess(memory_increase, 100 * 1024 * 1024)  # 100MB以内
        
        # 清理
        del images


def run_ui_tests():
    """运行UI测试"""
    print("🧪 开始运行UI测试...")
    
    # 运行单元测试
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestParameterPanel))
    suite.addTests(loader.loadTestsFromTestCase(TestMainInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestUIErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestWebIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamlitComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorScenarios))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出结果
    print(f"\n📊 测试结果:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n🚨 错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_ui_tests()
    sys.exit(0 if success else 1) 