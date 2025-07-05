#!/usr/bin/env python3
"""
HD Strategy 实现分析脚本
分析代码中HD策略的实现细节
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HDStrategyAnalyzer:
    """HD策略实现分析器"""
    
    def __init__(self):
        self.project_root = project_root
        self.analysis_results = {}
        
    def find_hd_strategy_references(self) -> Dict[str, List[str]]:
        """查找HD策略相关的代码引用"""
        hd_patterns = [
            r'hd_strategy',
            r'HDStrategy',
            r'HD_STRATEGY',
            r'ORIGINAL',
            r'CROP',
            r'RESIZE',
            r'crop_trigger',
            r'resize_limit',
            r'crop_margin'
        ]
        
        file_patterns = [
            '**/*.py',
            '**/*.yaml',
            '**/*.yml'
        ]
        
        references = {}
        
        for pattern in file_patterns:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        matches = []
                        for hd_pattern in hd_patterns:
                            if re.search(hd_pattern, content, re.IGNORECASE):
                                # 找到匹配的行
                                lines = content.split('\n')
                                for i, line in enumerate(lines, 1):
                                    if re.search(hd_pattern, line, re.IGNORECASE):
                                        matches.append(f"Line {i}: {line.strip()}")
                        
                        if matches:
                            references[str(file_path.relative_to(self.project_root))] = matches
                            
                    except Exception as e:
                        logger.warning(f"读取文件失败 {file_path}: {e}")
        
        return references
    
    def analyze_config_files(self) -> Dict[str, Any]:
        """分析配置文件中的HD策略设置"""
        config_files = [
            'config/config.py',
            'config/iopaint_config.yaml',
            'config/powerpaint_config.yaml',
            'web_config.yaml'
        ]
        
        config_analysis = {}
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 分析HD策略相关配置
                    hd_configs = []
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        if any(keyword in line.lower() for keyword in ['hd_strategy', 'crop', 'resize']):
                            hd_configs.append(f"Line {i}: {line.strip()}")
                    
                    if hd_configs:
                        config_analysis[config_file] = hd_configs
                        
                except Exception as e:
                    logger.warning(f"分析配置文件失败 {config_file}: {e}")
        
        return config_analysis
    
    def analyze_processor_implementation(self) -> Dict[str, Any]:
        """分析处理器中的HD策略实现"""
        processor_files = [
            'core/models/iopaint_processor.py',
            'core/models/lama_processor.py',
            'core/models/unified_processor.py'
        ]
        
        processor_analysis = {}
        
        for processor_file in processor_files:
            processor_path = self.project_root / processor_file
            if processor_path.exists():
                try:
                    with open(processor_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    analysis = {
                        'hd_strategy_usage': [],
                        'config_building': [],
                        'strategy_mapping': []
                    }
                    
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        line_lower = line.lower()
                        
                        if 'hd_strategy' in line_lower:
                            analysis['hd_strategy_usage'].append(f"Line {i}: {line.strip()}")
                        
                        if 'config' in line_lower and any(keyword in line_lower for keyword in ['crop', 'resize', 'original']):
                            analysis['config_building'].append(f"Line {i}: {line.strip()}")
                        
                        if any(keyword in line for keyword in ['CROP', 'RESIZE', 'ORIGINAL']):
                            analysis['strategy_mapping'].append(f"Line {i}: {line.strip()}")
                    
                    if any(analysis.values()):
                        processor_analysis[processor_file] = analysis
                        
                except Exception as e:
                    logger.warning(f"分析处理器文件失败 {processor_file}: {e}")
        
        return processor_analysis
    
    def check_iopaint_integration(self) -> Dict[str, Any]:
        """检查IOPaint集成中的HD策略"""
        integration_analysis = {}
        
        # 检查IOPaint是否正确安装
        try:
            from iopaint.schema import HDStrategy
            integration_analysis['iopaint_schema'] = {
                'installed': True,
                'hd_strategies': [attr for attr in dir(HDStrategy) if not attr.startswith('_')]
            }
        except ImportError as e:
            integration_analysis['iopaint_schema'] = {
                'installed': False,
                'error': str(e)
            }
        
        # 检查配置映射
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("iopaint_processor", 
                                                         self.project_root / "core/models/iopaint_processor.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 检查策略映射是否正确
            if hasattr(module, 'IOPaintProcessor'):
                integration_analysis['strategy_mapping'] = "Found IOPaintProcessor class"
            else:
                integration_analysis['strategy_mapping'] = "IOPaintProcessor class not found"
                
        except Exception as e:
            integration_analysis['import_error'] = str(e)
        
        return integration_analysis
    
    def generate_analysis_report(self) -> str:
        """生成分析报告"""
        logger.info("🔍 开始分析HD策略实现")
        
        # 收集分析数据
        references = self.find_hd_strategy_references()
        config_analysis = self.analyze_config_files()
        processor_analysis = self.analyze_processor_implementation()
        iopaint_integration = self.check_iopaint_integration()
        
        # 生成报告
        report = []
        report.append("=" * 80)
        report.append("HD Strategy 实现分析报告")
        report.append("=" * 80)
        report.append(f"分析时间: {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 代码引用分析
        report.append("📁 代码引用分析")
        report.append("-" * 40)
        if references:
            for file_path, matches in references.items():
                report.append(f"\n📄 {file_path}:")
                for match in matches:
                    report.append(f"  {match}")
        else:
            report.append("❌ 未找到HD策略相关代码引用")
        report.append("")
        
        # 配置文件分析
        report.append("⚙️  配置文件分析")
        report.append("-" * 40)
        if config_analysis:
            for config_file, configs in config_analysis.items():
                report.append(f"\n📄 {config_file}:")
                for config in configs:
                    report.append(f"  {config}")
        else:
            report.append("❌ 未找到HD策略配置")
        report.append("")
        
        # 处理器实现分析
        report.append("🔧 处理器实现分析")
        report.append("-" * 40)
        if processor_analysis:
            for processor_file, analysis in processor_analysis.items():
                report.append(f"\n📄 {processor_file}:")
                
                if analysis['hd_strategy_usage']:
                    report.append("  HD策略使用:")
                    for usage in analysis['hd_strategy_usage']:
                        report.append(f"    {usage}")
                
                if analysis['config_building']:
                    report.append("  配置构建:")
                    for config in analysis['config_building']:
                        report.append(f"    {config}")
                
                if analysis['strategy_mapping']:
                    report.append("  策略映射:")
                    for mapping in analysis['strategy_mapping']:
                        report.append(f"    {mapping}")
        else:
            report.append("❌ 未找到处理器中的HD策略实现")
        report.append("")
        
        # IOPaint集成分析
        report.append("🔗 IOPaint集成分析")
        report.append("-" * 40)
        if iopaint_integration:
            for key, value in iopaint_integration.items():
                if isinstance(value, dict):
                    report.append(f"{key}:")
                    for sub_key, sub_value in value.items():
                        report.append(f"  {sub_key}: {sub_value}")
                else:
                    report.append(f"{key}: {value}")
        report.append("")
        
        # 问题诊断
        report.append("🩺 问题诊断")
        report.append("-" * 40)
        
        issues = []
        
        # 检查IOPaint是否正确安装
        if not iopaint_integration.get('iopaint_schema', {}).get('installed', False):
            issues.append("❌ IOPaint未正确安装或导入失败")
        
        # 检查配置完整性
        required_configs = ['hd_strategy', 'crop_trigger', 'resize_limit']
        found_configs = set()
        for configs in config_analysis.values():
            for config in configs:
                for required in required_configs:
                    if required in config.lower():
                        found_configs.add(required)
        
        missing_configs = set(required_configs) - found_configs
        if missing_configs:
            issues.append(f"⚠️  缺少配置项: {', '.join(missing_configs)}")
        
        # 检查处理器实现
        if not processor_analysis:
            issues.append("❌ 未找到处理器中的HD策略实现")
        
        if issues:
            for issue in issues:
                report.append(issue)
        else:
            report.append("✅ 未发现明显问题")
        
        report.append("")
        
        # 建议
        report.append("💡 建议")
        report.append("-" * 40)
        report.append("1. 确保IOPaint正确安装: pip install iopaint")
        report.append("2. 验证HD策略配置参数完整性")
        report.append("3. 检查处理器中的策略映射是否正确")
        report.append("4. 运行测试脚本验证功能")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """主函数"""
    logger.info("🔍 开始HD策略实现分析")
    
    analyzer = HDStrategyAnalyzer()
    
    try:
        report = analyzer.generate_analysis_report()
        
        # 保存报告
        report_path = Path("scripts/hd_strategy_analysis_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 打印报告
        print(report)
        
        logger.info(f"📄 分析报告已保存到: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 分析过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)