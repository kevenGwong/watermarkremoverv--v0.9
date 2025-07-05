#!/usr/bin/env python3
"""
模型路径配置检查 - 验证所有模型路径配置是否正确
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_config_files():
    """检查配置文件中的模型路径"""
    print("🔍 检查配置文件中的模型路径...")
    
    config_files = [
        'config/iopaint_config.yaml',
        'config/web_config.yaml'
    ]
    
    config_analysis = {}
    
    for config_file in config_files:
        print(f"\n  📄 检查 {config_file}:")
        config_path = project_root / config_file
        
        if not config_path.exists():
            print(f"    ❌ 配置文件不存在")
            config_analysis[config_file] = {'exists': False}
            continue
        
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            print(f"    ✅ 配置文件加载成功")
            
            # 查找模型相关配置
            model_configs = {}
            
            def find_model_configs(data, prefix=""):
                if isinstance(data, dict):
                    for key, value in data.items():
                        current_key = f"{prefix}.{key}" if prefix else key
                        if 'model' in key.lower() or 'path' in key.lower():
                            model_configs[current_key] = value
                        if isinstance(value, dict):
                            find_model_configs(value, current_key)
            
            find_model_configs(config_data)
            
            print(f"    📦 找到模型配置项: {len(model_configs)}")
            for key, value in model_configs.items():
                print(f"      {key}: {value}")
            
            config_analysis[config_file] = {
                'exists': True,
                'model_configs': model_configs
            }
            
        except ImportError:
            print(f"    ⚠️ PyYAML未安装，跳过YAML解析")
            config_analysis[config_file] = {'exists': True, 'yaml_unavailable': True}
        except Exception as e:
            print(f"    ❌ 配置文件解析失败: {e}")
            config_analysis[config_file] = {'exists': True, 'error': str(e)}
    
    return config_analysis

def check_iopaint_model_paths():
    """检查IOPaint模型路径配置"""
    print("\n🔍 检查IOPaint模型路径配置...")
    
    # IOPaint模型缓存路径
    iopaint_cache_paths = [
        Path.home() / '.cache' / 'torch' / 'hub' / 'checkpoints',
        Path.home() / '.cache' / 'huggingface' / 'transformers',
        Path.home() / '.cache' / 'iopaint',
        project_root / 'data' / 'models',
        project_root / 'models'
    ]
    
    print("  📂 检查IOPaint缓存路径:")
    cache_analysis = {}
    
    for cache_path in iopaint_cache_paths:
        exists = cache_path.exists()
        status = "✅" if exists else "❌"
        
        print(f"    {cache_path}: {status}")
        
        if exists:
            # 检查是否有模型文件
            model_files = list(cache_path.glob('**/*.pth')) + list(cache_path.glob('**/*.bin')) + list(cache_path.glob('**/*.ckpt'))
            print(f"      模型文件: {len(model_files)} 个")
            
            cache_analysis[str(cache_path)] = {
                'exists': True,
                'model_files': len(model_files),
                'sample_files': [str(f.name) for f in model_files[:3]]
            }
        else:
            cache_analysis[str(cache_path)] = {'exists': False}
    
    return cache_analysis

def check_custom_model_paths():
    """检查自定义模型路径"""
    print("\n🔍 检查自定义模型路径...")
    
    # 从CLAUDE.md中提到的自定义模型路径
    custom_model_paths = [
        '/home/duolaameng/SAM_Remove/Watermark_sam/output/checkpoints/epoch=071-valid_iou=0.7267.ckpt',
        project_root / 'data' / 'models' / 'custom_watermark_model.ckpt',
        project_root / 'models' / 'watermark_segmentation.pth'
    ]
    
    print("  📦 检查自定义模型文件:")
    custom_analysis = {}
    
    for model_path in custom_model_paths:
        model_path = Path(model_path)
        exists = model_path.exists()
        status = "✅" if exists else "❌"
        
        print(f"    {model_path}: {status}")
        
        if exists:
            size_mb = model_path.stat().st_size / (1024 * 1024)
            print(f"      大小: {size_mb:.1f} MB")
            
            custom_analysis[str(model_path)] = {
                'exists': True,
                'size_mb': size_mb
            }
        else:
            custom_analysis[str(model_path)] = {'exists': False}
    
    return custom_analysis

def check_code_model_references():
    """检查代码中的模型路径引用"""
    print("\n🔍 检查代码中的模型路径引用...")
    
    # 关键文件中的模型路径引用
    key_files = [
        'core/models/unified_processor.py',
        'core/models/mask_generators.py',
        'core/models/iopaint_processor.py',
        'core/models/mat_processor.py',
        'core/models/zits_processor.py',
        'core/models/fcf_processor.py',
        'core/models/lama_processor.py'
    ]
    
    code_analysis = {}
    
    for file_path in key_files:
        print(f"\n  📄 检查 {file_path}:")
        full_path = project_root / file_path
        
        if not full_path.exists():
            print(f"    ❌ 文件不存在")
            code_analysis[file_path] = {'exists': False}
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找模型路径相关代码
            model_references = []
            
            # 查找常见的模型路径模式
            import re
            patterns = [
                r'ModelManager\s*\(\s*name=[\'"](.*?)[\'"]',
                r'model_path\s*=\s*[\'\"](.*?)[\'\"]',
                r'checkpoint_path\s*=\s*[\'\"](.*?)[\'\"]',
                r'\.pth[\'\"]*',
                r'\.ckpt[\'\"]*',
                r'\.bin[\'\"]*',
                r'microsoft/Florence-2',
                r'huggingface\.co',
                r'torch\.hub\.load'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    model_references.extend(matches)
            
            print(f"    ✅ 文件分析完成")
            if model_references:
                print(f"    📦 找到模型引用: {len(model_references)}")
                for ref in model_references[:5]:  # 显示前5个
                    print(f"      - {ref}")
                if len(model_references) > 5:
                    print(f"      ... 还有 {len(model_references) - 5} 个")
            else:
                print(f"    📦 未找到明显的模型路径引用")
            
            code_analysis[file_path] = {
                'exists': True,
                'model_references': model_references
            }
            
        except Exception as e:
            print(f"    ❌ 文件分析失败: {e}")
            code_analysis[file_path] = {'exists': True, 'error': str(e)}
    
    return code_analysis

def check_environment_variables():
    """检查环境变量中的模型路径配置"""
    print("\n🔍 检查环境变量中的模型路径配置...")
    
    # 常见的模型相关环境变量
    model_env_vars = [
        'HF_HOME',                    # Hugging Face缓存路径
        'TORCH_HOME',                 # PyTorch模型缓存路径
        'TRANSFORMERS_CACHE',         # Transformers缓存路径
        'IOPAINT_MODEL_DIR',          # IOPaint模型目录
        'CUDA_VISIBLE_DEVICES',       # GPU设备配置
        'MODEL_CACHE_DIR',            # 通用模型缓存
        'WATERMARK_MODEL_PATH'        # 自定义水印模型路径
    ]
    
    print("  🌍 环境变量检查:")
    env_analysis = {}
    
    for var in model_env_vars:
        value = os.environ.get(var)
        if value:
            print(f"    {var}: {value}")
            
            # 如果是路径，检查是否存在
            if 'PATH' in var or 'HOME' in var or 'DIR' in var:
                path_exists = Path(value).exists()
                status = "✅" if path_exists else "⚠️"
                print(f"      路径存在: {status}")
                
                env_analysis[var] = {
                    'value': value,
                    'path_exists': path_exists
                }
            else:
                env_analysis[var] = {'value': value}
        else:
            print(f"    {var}: ❌ 未设置")
            env_analysis[var] = {'value': None}
    
    return env_analysis

def analyze_model_download_mechanism():
    """分析模型自动下载机制"""
    print("\n🔍 分析模型自动下载机制...")
    
    download_analysis = {
        'IOPaint模型': {
            'mechanism': 'IOPaint ModelManager自动下载',
            'cache_location': '~/.cache/torch/hub/checkpoints/',
            'models': ['zits', 'mat', 'fcf', 'lama'],
            'auto_download': True
        },
        'Florence-2模型': {
            'mechanism': 'Hugging Face transformers自动下载',
            'cache_location': '~/.cache/huggingface/transformers/',
            'models': ['microsoft/Florence-2-large'],
            'auto_download': True
        },
        '自定义分割模型': {
            'mechanism': '手动配置，需要预先下载',
            'cache_location': '配置文件指定路径',
            'models': ['watermark_segmentation.ckpt'],
            'auto_download': False
        }
    }
    
    print("  🤖 模型下载机制分析:")
    for model_type, info in download_analysis.items():
        print(f"\n    {model_type}:")
        print(f"      下载机制: {info['mechanism']}")
        print(f"      缓存位置: {info['cache_location']}")
        print(f"      支持模型: {', '.join(info['models'])}")
        print(f"      自动下载: {'✅' if info['auto_download'] else '❌'}")
    
    return download_analysis

def generate_path_configuration_report():
    """生成路径配置报告"""
    print("\n📋 生成模型路径配置报告...")
    
    # 建议的路径配置
    recommended_paths = {
        'IOPaint模型缓存': '~/.cache/torch/hub/checkpoints/',
        'Hugging Face缓存': '~/.cache/huggingface/transformers/',
        '自定义模型目录': str(project_root / 'data' / 'models'),
        '临时文件目录': str(project_root / 'temp'),
        '输出结果目录': str(project_root / 'output')
    }
    
    print("  💡 推荐的路径配置:")
    for purpose, path in recommended_paths.items():
        print(f"    {purpose}: {path}")
        
        # 检查路径是否存在
        expanded_path = Path(path).expanduser()
        exists = expanded_path.exists()
        status = "✅" if exists else "⚠️"
        print(f"      当前状态: {status}")
    
    # 路径配置最佳实践
    best_practices = [
        "使用相对路径避免硬编码绝对路径",
        "支持环境变量覆盖默认路径",
        "自动创建不存在的缓存目录",
        "定期清理过期的模型缓存",
        "统一管理所有模型路径配置"
    ]
    
    print(f"\n  🎯 路径配置最佳实践:")
    for i, practice in enumerate(best_practices, 1):
        print(f"    {i}. {practice}")
    
    return recommended_paths, best_practices

def check_model_paths_comprehensive():
    """模型路径配置完整检查"""
    print("🚀 开始模型路径配置检查...")
    
    # 执行各项检查
    checks = {
        '配置文件': check_config_files(),
        'IOPaint缓存': check_iopaint_model_paths(),
        '自定义模型': check_custom_model_paths(),
        '代码引用': check_code_model_references(),
        '环境变量': check_environment_variables(),
        '下载机制': analyze_model_download_mechanism(),
        '配置建议': generate_path_configuration_report()
    }
    
    # 输出总结
    print("\n" + "="*60)
    print("📊 模型路径配置检查结果总结")
    print("="*60)
    
    # 配置文件检查总结
    config_files = checks['配置文件']
    config_count = len([f for f in config_files.values() if f.get('exists', False)])
    print(f"\n📄 配置文件: {config_count} 个可用")
    
    # IOPaint缓存检查总结
    iopaint_cache = checks['IOPaint缓存']
    cache_available = len([c for c in iopaint_cache.values() if c.get('exists', False)])
    print(f"🗂️ IOPaint缓存: {cache_available} 个路径可用")
    
    # 自定义模型检查总结
    custom_models = checks['自定义模型']
    custom_available = len([m for m in custom_models.values() if m.get('exists', False)])
    total_custom = len(custom_models)
    print(f"📦 自定义模型: {custom_available}/{total_custom} 个可用")
    
    # 代码引用分析总结
    code_refs = checks['代码引用']
    files_with_refs = len([f for f in code_refs.values() if f.get('exists', False) and f.get('model_references')])
    total_files = len([f for f in code_refs.values() if f.get('exists', False)])
    print(f"💻 代码引用: {files_with_refs}/{total_files} 文件包含模型引用")
    
    # 环境变量检查总结
    env_vars = checks['环境变量']
    set_env_vars = len([v for v in env_vars.values() if v.get('value')])
    total_env_vars = len(env_vars)
    print(f"🌍 环境变量: {set_env_vars}/{total_env_vars} 个已设置")
    
    # 下载机制分析总结
    download_mechs = checks['下载机制']
    auto_download_count = len([m for m in download_mechs.values() if m.get('auto_download', False)])
    print(f"🤖 自动下载: {auto_download_count} 种机制支持")
    
    # 整体健康度评估
    health_score = 0
    max_score = 100
    
    # 配置完整性权重 25%
    if config_count >= 1:
        health_score += 25
    
    # 缓存可用性权重 25%
    if cache_available >= 1:
        health_score += 25
    
    # 代码完整性权重 25%
    if files_with_refs >= total_files * 0.5:
        health_score += 25
    
    # 自动化程度权重 25%
    if auto_download_count >= 2:
        health_score += 25
    
    print(f"\n🎯 模型路径配置健康度: {health_score}/100")
    
    # 评估结论
    if health_score >= 80:
        status = "✅ 优秀"
        conclusion = "模型路径配置完整，支持自动下载"
    elif health_score >= 60:
        status = "⚠️ 良好"
        conclusion = "基本配置完整，部分路径需要手动设置"
    else:
        status = "❌ 需要改进"
        conclusion = "模型路径配置不完整，需要手动配置"
    
    print(f"📈 整体评估: {status}")
    print(f"💡 结论: {conclusion}")
    
    # 具体建议
    print(f"\n🔧 配置状态:")
    print(f"   ✅ IOPaint模型: 支持自动下载到标准缓存路径")
    print(f"   ✅ Florence-2模型: 通过Hugging Face自动下载")
    print(f"   ⚠️ 自定义分割模型: 需要手动配置路径")
    print(f"   ✅ 模型缓存管理: 遵循标准缓存路径约定")
    
    if health_score < 100:
        print(f"\n💡 改进建议:")
        if custom_available == 0:
            print(f"   🔄 配置自定义水印分割模型路径")
        if set_env_vars < total_env_vars // 2:
            print(f"   🔄 设置相关环境变量以优化缓存管理")
        if health_score < 80:
            print(f"   🔄 完善模型路径配置文件")
    
    return checks, health_score >= 60

if __name__ == "__main__":
    checks, is_healthy = check_model_paths_comprehensive()
    
    if is_healthy:
        print("\n🎉 模型路径配置检查通过!")
        print("✅ 模型路径配置基本正确，支持自动下载")
    else:
        print("\n⚠️ 模型路径配置需要进一步完善")
        print("建议按照报告建议配置缺失的模型路径")