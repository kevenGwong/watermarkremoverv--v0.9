"""
参数控制面板组件
"""

import streamlit as st
from typing import Dict, Any

class ParameterPanel:
    """参数控制面板"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """初始化session state"""
        if 'processor' not in st.session_state:
            st.session_state.processor = None
        if 'processing_result' not in st.session_state:
            st.session_state.processing_result = None
        if 'original_image' not in st.session_state:
            st.session_state.original_image = None
    
    def render(self) -> Dict[str, Any]:
        """渲染参数面板"""
        st.sidebar.title("🔬 参数控制")
        
        # 模型选择
        mask_model = st.sidebar.selectbox(
            "Mask生成方法",
            ["custom", "florence2", "upload"],
            format_func=lambda x: {
                "custom": "自定义分割模型",
                "florence2": "Florence-2检测",
                "upload": "上传自定义Mask"
            }[x]
        )
        
        # 根据模型显示不同参数
        mask_params = self._render_mask_params(mask_model)
        inpaint_params = self._render_inpaint_params()
        performance_params = self._render_performance_params()
        
        # 处理按钮
        if st.sidebar.button("🚀 开始处理", type="primary"):
            return {
                "mask_model": mask_model,
                "mask_params": mask_params,
                "inpaint_params": inpaint_params,
                "performance_params": performance_params
            }
        
        return {}
    
    def _render_mask_params(self, mask_model: str) -> Dict[str, Any]:
        """渲染mask参数"""
        st.sidebar.subheader("🎯 Mask参数")
        params = {}
        
        if mask_model == "custom":
            params['mask_threshold'] = st.sidebar.slider(
                "分割阈值", 0.0, 1.0, 0.5, 0.01
            )
            params['mask_dilate_kernel_size'] = st.sidebar.slider(
                "膨胀核大小", 1, 50, 3, 1
            )
            params['mask_dilate_iterations'] = st.sidebar.slider(
                "膨胀迭代次数", 1, 20, 1, 1
            )
        
        elif mask_model == "florence2":
            params['max_bbox_percent'] = st.sidebar.slider(
                "最大边界框比例", 1.0, 50.0, 10.0, 0.5
            )
            params['detection_prompt'] = st.sidebar.text_input(
                "检测提示词", "watermark"
            )
        
        elif mask_model == "upload":
            uploaded_mask = st.sidebar.file_uploader(
                "上传Mask文件", type=['png', 'jpg', 'jpeg']
            )
            params['uploaded_mask'] = uploaded_mask
            params['mask_dilate_kernel_size'] = st.sidebar.slider(
                "额外膨胀核大小", 0, 20, 0, 1
            )
        
        return params
    
    def _render_inpaint_params(self) -> Dict[str, Any]:
        """渲染修复参数"""
        st.sidebar.subheader("🎨 修复参数")
        
        return {
            'ldm_steps': st.sidebar.slider("LDM步数", 10, 200, 50, 5),
            'ldm_sampler': st.sidebar.selectbox("LDM采样器", ["ddim", "plms"]),
            'hd_strategy': st.sidebar.selectbox(
                "高清策略",
                ["CROP", "RESIZE", "ORIGINAL"],
                format_func=lambda x: {
                    "CROP": "裁剪策略",
                    "RESIZE": "缩放策略", 
                    "ORIGINAL": "原始策略"
                }[x]
            )
        }
    
    def _render_performance_params(self) -> Dict[str, Any]:
        """渲染性能参数"""
        st.sidebar.subheader("⚡ 性能参数")
        
        return {
            'mixed_precision': st.sidebar.checkbox("混合精度", value=False),
            'device': st.sidebar.selectbox("设备", ["auto", "cpu", "cuda"])
        } 