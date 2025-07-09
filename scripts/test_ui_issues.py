#!/usr/bin/env python3
"""
UI问题检测脚本
专门检测Web UI中的常见问题，如参数错误、方法调用错误等
"""

import sys
import os
import ast
import re
from pathlib import Path

def check_python_syntax(file_path):
    """检查Python文件语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"解析错误: {e}"

def check_method_signatures(file_path):
    """检查方法签名一致性"""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析AST
        tree = ast.parse(content)
        
        # 查找方法定义和调用
        method_defs = {}
        method_calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # 记录方法定义
                args = [arg.arg for arg in node.args.args]
                if node.name.startswith('_'):  # 私有方法
                    method_defs[node.name] = len(args)
            
            elif isinstance(node, ast.Call):
                # 记录方法调用
                if isinstance(node.func, ast.Attribute):
                    method_calls.append((node.func.attr, len(node.args)))
        
        # 检查调用和定义是否匹配
        for method_name, call_args in method_calls:
            if method_name in method_defs:
                def_args = method_defs[method_name]
                if call_args != def_args:
                    issues.append(f"方法 '{method_name}' 参数不匹配: 定义 {def_args} 个，调用 {call_args} 个")
    
    except Exception as e:
        issues.append(f"检查方法签名时出错: {e}")
    
    return issues

def check_streamlit_components(file_path):
    """检查Streamlit组件使用"""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查重复的key
        key_pattern = r'key=["\']([^"\']+)["\']'
        keys = re.findall(key_pattern, content)
        duplicate_keys = [key for key in set(keys) if keys.count(key) > 1]
        
        if duplicate_keys:
            issues.append(f"发现重复的Streamlit key: {duplicate_keys}")
        
        # 检查selectbox选项
        selectbox_pattern = r'st\.selectbox\([^)]*options=\[([^\]]+)\][^)]*\)'
        selectbox_matches = re.findall(selectbox_pattern, content)
        
        for match in selectbox_matches:
            if 'auto' in match.lower():
                issues.append("发现可能不需要的'auto'选项")
        
        # 检查按钮处理
        button_pattern = r'if st\.button\([^)]*\):'
        if not re.search(button_pattern, content):
            issues.append("可能缺少处理按钮的逻辑")
    
    except Exception as e:
        issues.append(f"检查Streamlit组件时出错: {e}")
    
    return issues

def check_imports(file_path):
    """检查导入问题"""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查相对导入
        relative_imports = re.findall(r'from \.([^ ]+) import', content)
        if relative_imports:
            issues.append(f"发现相对导入: {relative_imports}")
        
        # 检查可能缺失的导入
        if 'streamlit' in content and 'import streamlit' not in content:
            issues.append("可能缺失streamlit导入")
        
        if 'PIL' in content and 'from PIL' not in content:
            issues.append("可能缺失PIL导入")
    
    except Exception as e:
        issues.append(f"检查导入时出错: {e}")
    
    return issues

def check_parameter_handling(file_path):
    """检查参数处理"""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查参数解包
        unpack_pattern = r'(\w+), (\w+), (\w+), (\w+), (\w+) ='
        unpack_matches = re.findall(unpack_pattern, content)
        
        for match in unpack_matches:
            # 检查是否所有变量都被使用
            for var in match:
                if content.count(var) < 2:  # 只出现一次，可能未使用
                    issues.append(f"参数 '{var}' 可能未被使用")
        
        # 检查字典访问
        dict_access_pattern = r'(\w+)\.get\(([^)]+)\)'
        dict_matches = re.findall(dict_access_pattern, content)
        
        for var, key in dict_matches:
            if 'default' not in content and 'None' not in content:
                issues.append(f"字典访问 '{var}.get({key})' 可能缺少默认值")
    
    except Exception as e:
        issues.append(f"检查参数处理时出错: {e}")
    
    return issues

def check_error_handling(file_path):
    """检查错误处理"""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查异常处理
        if 'try:' in content and 'except' not in content:
            issues.append("发现try块但缺少except处理")
        
        if 'except:' in content and 'Exception' not in content:
            issues.append("建议使用具体的异常类型而不是通用except")
        
        # 检查错误消息
        error_pattern = r'st\.error\([^)]*\)'
        if not re.search(error_pattern, content):
            issues.append("可能缺少错误显示逻辑")
    
    except Exception as e:
        issues.append(f"检查错误处理时出错: {e}")
    
    return issues

def analyze_ui_file(file_path):
    """分析UI文件"""
    print(f"\n🔍 分析文件: {file_path}")
    print("-" * 50)
    
    all_issues = []
    
    # 检查语法
    syntax_ok, syntax_error = check_python_syntax(file_path)
    if not syntax_ok:
        all_issues.append(("语法错误", [syntax_error]))
        print(f"❌ 语法错误: {syntax_error}")
        return all_issues
    else:
        print("✅ 语法检查通过")
    
    # 检查方法签名
    method_issues = check_method_signatures(file_path)
    if method_issues:
        all_issues.append(("方法签名", method_issues))
        print(f"⚠️ 发现 {len(method_issues)} 个方法签名问题")
        for issue in method_issues:
            print(f"   - {issue}")
    else:
        print("✅ 方法签名检查通过")
    
    # 检查Streamlit组件
    streamlit_issues = check_streamlit_components(file_path)
    if streamlit_issues:
        all_issues.append(("Streamlit组件", streamlit_issues))
        print(f"⚠️ 发现 {len(streamlit_issues)} 个Streamlit组件问题")
        for issue in streamlit_issues:
            print(f"   - {issue}")
    else:
        print("✅ Streamlit组件检查通过")
    
    # 检查导入
    import_issues = check_imports(file_path)
    if import_issues:
        all_issues.append(("导入问题", import_issues))
        print(f"⚠️ 发现 {len(import_issues)} 个导入问题")
        for issue in import_issues:
            print(f"   - {issue}")
    else:
        print("✅ 导入检查通过")
    
    # 检查参数处理
    param_issues = check_parameter_handling(file_path)
    if param_issues:
        all_issues.append(("参数处理", param_issues))
        print(f"⚠️ 发现 {len(param_issues)} 个参数处理问题")
        for issue in param_issues:
            print(f"   - {issue}")
    else:
        print("✅ 参数处理检查通过")
    
    # 检查错误处理
    error_issues = check_error_handling(file_path)
    if error_issues:
        all_issues.append(("错误处理", error_issues))
        print(f"⚠️ 发现 {len(error_issues)} 个错误处理问题")
        for issue in error_issues:
            print(f"   - {issue}")
    else:
        print("✅ 错误处理检查通过")
    
    return all_issues

def main():
    """主函数"""
    print("🔍 UI问题检测工具")
    print("=" * 60)
    
    # 要检查的文件
    ui_files = [
        'interfaces/web/main.py',
        'interfaces/web/ui.py'
    ]
    
    total_issues = 0
    all_results = {}
    
    for file_path in ui_files:
        if os.path.exists(file_path):
            issues = analyze_ui_file(file_path)
            all_results[file_path] = issues
            total_issues += sum(len(issue_list) for _, issue_list in issues)
        else:
            print(f"❌ 文件不存在: {file_path}")
    
    # 总结报告
    print("\n" + "=" * 60)
    print("📊 检测报告")
    print("=" * 60)
    
    if total_issues == 0:
        print("🎉 没有发现明显问题！")
    else:
        print(f"⚠️ 发现 {total_issues} 个潜在问题:")
        
        for file_path, issues in all_results.items():
            if issues:
                print(f"\n📁 {file_path}:")
                for category, issue_list in issues:
                    print(f"  {category}:")
                    for issue in issue_list:
                        print(f"    - {issue}")
    
    print(f"\n⏱️ 检测完成，共检查 {len(ui_files)} 个文件")
    
    return total_issues == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 