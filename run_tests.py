#!/usr/bin/env python3
"""
自动化测试运行脚本
一键运行所有测试，包括单元测试、集成测试和端到端测试
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))

def run_command(cmd, description, timeout=300):
    """运行命令并返回结果"""
    print(f"\n🚀 {description}")
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.path.dirname(__file__)
        )
        
        if result.returncode == 0:
            print(f"✅ {description} 成功")
            if result.stdout:
                print("输出:", result.stdout[-500:])  # 只显示最后500字符
            return True
        else:
            print(f"❌ {description} 失败")
            print("错误输出:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} 超时")
        return False
    except Exception as e:
        print(f"💥 {description} 异常: {e}")
        return False

def run_unit_tests():
    """运行单元测试"""
    return run_command(
        [sys.executable, '-m', 'pytest', 'tests/test_ui_components.py', '-v'],
        "单元测试"
    )

def run_integration_tests():
    """运行集成测试"""
    return run_command(
        [sys.executable, 'tests/test_web_integration.py'],
        "集成测试"
    )

def run_e2e_tests():
    """运行端到端测试"""
    return run_command(
        [sys.executable, 'tests/test_e2e_workflow.py'],
        "端到端测试"
    )

def run_linting():
    """运行代码检查"""
    return run_command(
        [sys.executable, '-m', 'flake8', 'interfaces/web/', '--max-line-length=120'],
        "代码风格检查"
    )

def run_type_checking():
    """运行类型检查"""
    try:
        import mypy
        return run_command(
            [sys.executable, '-m', 'mypy', 'interfaces/web/', '--ignore-missing-imports'],
            "类型检查"
        )
    except ImportError:
        print("⚠️ mypy未安装，跳过类型检查")
        return True

def check_dependencies():
    """检查依赖"""
    print("\n🔍 检查依赖...")
    
    required_packages = [
        'streamlit',
        'PIL',
        'numpy',
        'torch',
        'iopaint'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ 缺失依赖: {', '.join(missing_packages)}")
        return False
    else:
        print("✅ 所有依赖已安装")
        return True

def check_config_files():
    """检查配置文件"""
    print("\n📋 检查配置文件...")
    
    config_files = [
        'config/unified_config.yaml',
        'interfaces/web/main.py',
        'interfaces/web/ui.py'
    ]
    
    missing_files = []
    for file_path in config_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ 缺失文件: {', '.join(missing_files)}")
        return False
    else:
        print("✅ 所有配置文件存在")
        return True

def run_quick_tests():
    """运行快速测试（不包含端到端测试）"""
    print("🧪 运行快速测试套件...")
    
    tests = [
        ("依赖检查", check_dependencies),
        ("配置文件检查", check_config_files),
        ("代码风格检查", run_linting),
        ("单元测试", run_unit_tests),
        ("集成测试", run_integration_tests)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    return results

def run_full_tests():
    """运行完整测试套件"""
    print("🧪 运行完整测试套件...")
    
    tests = [
        ("依赖检查", check_dependencies),
        ("配置文件检查", check_config_files),
        ("代码风格检查", run_linting),
        ("类型检查", run_type_checking),
        ("单元测试", run_unit_tests),
        ("集成测试", run_integration_tests),
        ("端到端测试", run_e2e_tests)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    return results

def generate_test_report(results):
    """生成测试报告"""
    print("\n" + "="*60)
    print("📊 测试报告")
    print("="*60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("-"*60)
    print(f"总计: {passed + failed} 测试")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"成功率: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        return True
    else:
        print(f"\n💥 {failed} 个测试失败")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行WatermarkRemover-AI测试套件")
    parser.add_argument(
        '--mode', 
        choices=['quick', 'full', 'unit', 'integration', 'e2e'],
        default='quick',
        help='测试模式 (默认: quick)'
    )
    parser.add_argument(
        '--report', 
        action='store_true',
        help='生成详细报告'
    )
    
    args = parser.parse_args()
    
    print("🧪 WatermarkRemover-AI 测试套件")
    print("="*60)
    
    start_time = time.time()
    
    if args.mode == 'quick':
        results = run_quick_tests()
    elif args.mode == 'full':
        results = run_full_tests()
    elif args.mode == 'unit':
        results = [("单元测试", run_unit_tests())]
    elif args.mode == 'integration':
        results = [("集成测试", run_integration_tests())]
    elif args.mode == 'e2e':
        results = [("端到端测试", run_e2e_tests())]
    
    end_time = time.time()
    
    # 生成报告
    success = generate_test_report(results)
    
    print(f"\n⏱️ 总耗时: {end_time - start_time:.2f}秒")
    
    # 退出码
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 