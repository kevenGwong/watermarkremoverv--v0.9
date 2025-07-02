# 🎨 AI Watermark Remover

**Advanced AI-Powered Watermark Removal Tool with Debug UI for Parameter Control**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/licenses/)
[![Version: v1.0](https://img.shields.io/badge/Version-v1.0-green.svg)]()

## 🔬 **Current Best Version: Debug UI Edition**

The Debug Edition provides the most comprehensive parameter control and real-time comparison features for optimal watermark removal results.

**Latest Update (2025-07-02)**: ✅ **Major Quality Improvements**
- Fixed color channel issues (BGR/RGB conversion)
- Fixed mask loading problems 
- Enhanced parameter passing system
- Achieved remwm.py equivalent quality

![Debug UI Features](https://github.com/user-attachments/assets/8f7fb600-695f-4dd7-958c-0cff516b5c7a)

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
conda activate py310aiwatermark
```

### 2. Launch Debug UI
```bash
./run_debug_app.sh
```

### 3. Access Web Interface
```
http://localhost:8506
```

---

## ✨ Features

### 🔬 **Debug UI Edition (Recommended)**
- **📊 Left Panel**: Complete parameter control (30+ settings)
- **🔄 Right Panel**: Interactive before/after comparison
- **🎯 Dual Model Support**: Custom watermark detection + Florence-2
- **⚙️ Advanced Parameters**: Real-time adjustment with immediate feedback
- **🎭 Mask Visualization**: See exactly what the AI detects
- **💾 Multiple Export Formats**: PNG, WEBP, JPG with optimization

### 🎯 **Core AI Capabilities**
- **Custom Watermark Model**: FPN + MIT-B5 specialized for watermark detection
- **Florence-2 Integration**: Microsoft's multimodal model for text-guided detection  
- **LaMA Inpainting**: State-of-the-art context-aware filling
- **Transparency Mode**: Create transparent regions instead of filling

### 🔧 **Parameter Control**
- **Mask Generation**: Threshold, dilation, iterations (1-50 kernel, 1-20 iterations)
- **Detection Control**: Prompts, confidence, bbox limits
- **Inpainting Settings**: Steps, samplers, HD strategies
- **Performance Options**: Mixed precision, logging, seeds

---

## 📁 Directory Structure

### 🎯 Core Files
- **`watermark_web_app_debug.py`** - 🔬 Main Debug UI application
- **`run_debug_app.sh`** - 🚀 Launch script
- **`web_backend.py`** - ⚙️ Core processing backend
- **`web_config.yaml`** - 📋 Configuration
- **`utils.py`** - 🛠️ Utility functions

### 📚 Documentation
- **`DEBUG_UI_GUIDE.md`** - 🔬 Complete debug UI guide
- **`PARAMETER_RANGES_UPDATE.md`** - 📊 Parameter ranges explained
- **`DIRECTORY_STRUCTURE.md`** - 📁 Full directory reference

### 📦 Scripts Archive
- **`scripts/`** - 🗃️ Test scripts, deprecated versions, and documentation archive
  - `test_scripts/` - Testing and validation scripts
  - `deprecated_apps/` - Previous UI versions
  - `test_outputs/` - Test results and comparisons
  - `docs/` - Historical documentation

---

## 🎯 Usage Guide

### 🔬 Debug UI Workflow

1. **📸 Upload Image**: Choose image with watermarks
2. **🎯 Select Model**: Custom watermark or Florence-2 detection
3. **⚙️ Adjust Parameters**: Real-time parameter control in left panel
4. **🚀 Process**: Click process and watch real-time progress
5. **🔄 Compare**: Interactive slider shows before/after
6. **💾 Download**: Multiple format options with optimization

### 📊 Parameter Tuning Strategy

#### 🎯 Mask Detection
```
Start: threshold=0.5, kernel_size=3, iterations=1
↓
Insufficient detection: Lower threshold (0.3-0.4)
↓  
Too much detection: Raise threshold (0.6-0.7)
↓
Boundary issues: Adjust kernel_size (3-15)
↓
Coverage problems: Increase iterations (2-5)
↓
Complex watermarks: Use larger values (15-50, 5-20)
```

#### 🎨 Quality vs Speed
```
Speed Priority: steps=20-50, sampler=plms, strategy=RESIZE
↓
Balanced: steps=50-100, sampler=ddim, strategy=CROP  
↓
Quality Priority: steps=100-200, sampler=ddim, HD strategy
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 4GB+ GPU VRAM

### Setup
```bash
# Clone repository
git clone [repository-url]
cd WatermarkRemover-AI

# Run setup script
./setup.sh

# Activate environment
conda activate py310aiwatermark

# Launch debug UI
./run_debug_app.sh
```

---

## 🎨 Alternative Interfaces

### 🖥️ Original CLI/GUI
```bash
# Command line
python remwm.py input.jpg

# PyQt6 GUI  
python remwmgui.py
```

### 📱 Previous Web Versions
Historical web interface versions are available in `scripts/deprecated_apps/` for reference.

---

## 🔧 Technical Details

### 🤖 AI Models
- **Custom Watermark Model**: FPN + MIT-B5 encoder trained specifically for watermark segmentation
- **Florence-2**: Microsoft's multimodal foundation model for open-vocabulary detection
- **LaMA**: Large Mask Inpainting for high-quality context-aware filling

### ⚡ Performance
- **GPU Acceleration**: CUDA support for all models
- **Memory Optimization**: Efficient processing for large images
- **Batch Processing**: Multiple image support
- **Mixed Precision**: Faster processing with maintained quality

### 📊 Supported Formats
- **Input**: JPG, PNG, WEBP, TIFF, BMP
- **Output**: PNG (transparency), WEBP (compression), JPG (compatibility)

---

## 📈 Development History

This project evolved through multiple iterations:
1. **CLI Version** - Original command-line tool
2. **GUI Version** - PyQt6 desktop interface  
3. **Web Versions** - Multiple Streamlit iterations
4. **Debug Edition** - Current best version with full parameter control

All previous versions are preserved in `scripts/deprecated_apps/` for reference and comparison.

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎉 Acknowledgments

- **Florence-2**: Microsoft Research for the multimodal foundation model
- **LaMA**: Samsung AI for the inpainting model
- **IOPaint**: For the excellent inpainting pipeline
- **Streamlit**: For the web framework enabling rapid UI development

---

**🔬 Ready to remove watermarks with precision? Start with the Debug UI!**

```bash
conda activate py310aiwatermark
./run_debug_app.sh
# → http://localhost:8506
```