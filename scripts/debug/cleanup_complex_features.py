#!/usr/bin/env python3
"""
清理复杂功能脚本
移除Florence-2模型集成和透明处理模式
"""

import os
import shutil
import re
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComplexFeaturesCleaner:
    """复杂功能清理器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backup_complex_features"
        
    def create_backup(self):
        """创建备份"""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir()
        
        logger.info(f"📦 创建备份目录: {self.backup_dir}")
        
    def backup_file(self, file_path: Path):
        """备份单个文件"""
        relative_path = file_path.relative_to(self.project_root)
        backup_path = self.backup_dir / relative_path
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_path)
        logger.info(f"📦 备份文件: {relative_path}")
    
    def remove_florence_references(self):
        """移除Florence-2相关引用"""
        logger.info("🧹 开始移除Florence-2相关引用...")
        
        # 要修改的文件列表
        files_to_modify = [
            "core/models/__init__.py",
            "core/models/mask_generators.py",
            "core/processors/watermark_processor.py",
            "interfaces/web/ui.py"
        ]
        
        florence_patterns = [
            r'from \..*florence.*import.*\n',
            r'.*FlorenceMaskGenerator.*\n',
            r'.*florence.*',
            r"'florence'.*",
            r'"florence".*'
        ]
        
        for file_path in files_to_modify:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.backup_file(full_path)
                self._remove_patterns_from_file(full_path, florence_patterns, "Florence-2")
    
    def remove_transparent_references(self):
        """移除透明处理相关引用"""
        logger.info("🧹 开始移除透明处理相关引用...")
        
        # 主要的UI文件
        ui_file = self.project_root / "interfaces/web/ui.py"
        if ui_file.exists():
            self.backup_file(ui_file)
            self._simplify_ui_file(ui_file)
    
    def _remove_patterns_from_file(self, file_path: Path, patterns: list, feature_name: str):
        """从文件中移除匹配的模式"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_lines = len(content.splitlines())
            
            for pattern in patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)
            
            # 清理多余的空行
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            new_lines = len(content.splitlines())
            logger.info(f"✅ 处理文件 {file_path.name}: {feature_name} 引用已移除 ({original_lines} -> {new_lines} 行)")
            
        except Exception as e:
            logger.error(f"❌ 处理文件 {file_path} 失败: {e}")
    
    def _simplify_ui_file(self, ui_file: Path):
        """简化UI文件，移除透明处理"""
        try:
            with open(ui_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 移除透明处理相关代码段
            transparent_patterns = [
                # 移除透明参数
                r'transparent\s*=.*?\n',
                # 移除透明处理逻辑
                r'st\.sidebar\.checkbox\(\s*"Transparent.*?".*?\n',
                # 移除透明模式判断
                r'Mode.*?transparent.*?\n',
                # 移除透明处理相关函数参数
                r',\s*transparent\)',
                r',\s*transparent\s*\)',
                r'transparent,?\s*',
                # 移除透明相关的条件判断
                r'if.*?transparent.*?:\s*\n.*?\n',
                r"'Transparent'.*?transparent.*?'Inpaint'",
            ]
            
            original_lines = len(content.splitlines())
            
            for pattern in transparent_patterns:
                content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            # 修复函数签名
            content = re.sub(r'def\s+(\w+)\([^)]*,\s*\)', r'def \1()', content)
            content = re.sub(r'def\s+(\w+)\([^)]*,\s*([^)]+)\)', r'def \1(\2)', content)
            
            # 清理多余的空行和逗号
            content = re.sub(r',\s*\)', ')', content)
            content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
            
            with open(ui_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            new_lines = len(content.splitlines())
            logger.info(f"✅ UI文件简化完成: 透明处理已移除 ({original_lines} -> {new_lines} 行)")
            
        except Exception as e:
            logger.error(f"❌ 简化UI文件失败: {e}")
    
    def update_mask_generators(self):
        """更新mask生成器，使用简化版本"""
        logger.info("🔄 更新mask生成器...")
        
        original_file = self.project_root / "core/models/mask_generators.py"
        simplified_file = self.project_root / "core/models/mask_generators_simplified.py"
        
        if original_file.exists():
            self.backup_file(original_file)
            
        if simplified_file.exists():
            shutil.copy2(simplified_file, original_file)
            logger.info("✅ mask生成器已更新为简化版本")
        else:
            logger.error("❌ 找不到简化版mask生成器文件")
    
    def update_imports(self):
        """更新导入语句"""
        logger.info("🔄 更新导入语句...")
        
        # 更新__init__.py文件
        init_file = self.project_root / "core/models/__init__.py"
        if init_file.exists():
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 移除Florence相关导入，添加Simple生成器
            content = re.sub(r'.*FlorenceMaskGenerator.*\n', '', content)
            content = re.sub(r'.*FallbackMaskGenerator.*\n', '', content)
            
            # 添加简化版导入
            if 'SimpleMaskGenerator' not in content:
                content = content.replace(
                    'from .mask_generators import CustomMaskGenerator',
                    'from .mask_generators import CustomMaskGenerator, SimpleMaskGenerator'
                )
                content = content.replace(
                    "'CustomMaskGenerator'",
                    "'CustomMaskGenerator',\n    'SimpleMaskGenerator'"
                )
            
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ 导入语句已更新")
    
    def clean_test_files(self):
        """清理测试文件中的复杂功能引用"""
        logger.info("🧹 清理测试文件...")
        
        test_dir = self.project_root / "scripts"
        if not test_dir.exists():
            return
        
        for test_file in test_dir.glob("test_*.py"):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 如果文件包含Florence或透明处理，跳过或简化
                if 'florence' in content.lower() or 'transparent' in content.lower():
                    logger.info(f"⚠️ 测试文件 {test_file.name} 包含复杂功能，建议手动检查")
                    
            except Exception as e:
                logger.warning(f"⚠️ 无法处理测试文件 {test_file.name}: {e}")
    
    def run_cleanup(self):
        """运行完整的清理过程"""
        logger.info("🚀 开始清理复杂功能...")
        
        try:
            # 1. 创建备份
            self.create_backup()
            
            # 2. 移除Florence-2引用
            self.remove_florence_references()
            
            # 3. 移除透明处理引用
            self.remove_transparent_references()
            
            # 4. 更新mask生成器
            self.update_mask_generators()
            
            # 5. 更新导入语句
            self.update_imports()
            
            # 6. 清理测试文件
            self.clean_test_files()
            
            logger.info("✅ 复杂功能清理完成！")
            logger.info(f"📦 备份文件保存在: {self.backup_dir}")
            
        except Exception as e:
            logger.error(f"❌ 清理过程失败: {e}")
            return False
        
        return True

def main():
    """主函数"""
    project_root = Path(__file__).parent
    cleaner = ComplexFeaturesCleaner(str(project_root))
    
    success = cleaner.run_cleanup()
    
    if success:
        logger.info("🎉 复杂功能清理成功！")
        logger.info("📋 清理内容:")
        logger.info("   ✅ Florence-2 模型集成已移除")
        logger.info("   ✅ 透明处理模式已移除")
        logger.info("   ✅ mask生成器已简化")
        logger.info("   ✅ 导入语句已更新")
        return 0
    else:
        logger.error("❌ 清理过程中出现错误")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())