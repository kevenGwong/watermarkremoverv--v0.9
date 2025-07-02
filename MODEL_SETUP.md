# 模型文件设置说明

## 问题说明

由于模型文件 `models/epoch=071-valid_iou=0.7267.ckpt` 大小超过1GB，超过了GitHub的100MB文件大小限制，因此该文件没有被包含在Git仓库中。

## 解决方案

### 方法1：手动下载模型文件

1. 在项目根目录创建 `models/` 文件夹：
   ```bash
   mkdir models
   ```

2. 将模型文件 `epoch=071-valid_iou=0.7267.ckpt` 放入 `models/` 目录中。

### 方法2：使用Git LFS（推荐）

如果您想要将模型文件也进行版本控制，建议使用Git Large File Storage (LFS)：

1. 安装Git LFS：
   ```bash
   # Ubuntu/Debian
   sudo apt install git-lfs
   
   # 或者使用conda
   conda install git-lfs
   ```

2. 在项目中初始化LFS：
   ```bash
   git lfs install
   git lfs track "*.ckpt"
   git add .gitattributes
   git commit -m "Add LFS tracking for model files"
   ```

3. 添加模型文件：
   ```bash
   git add models/epoch=071-valid_iou=0.7267.ckpt
   git commit -m "Add model file via LFS"
   git push
   ```

## 当前状态

- ✅ 代码已成功推送到GitHub
- ✅ .gitignore已配置，排除大文件
- ⚠️ 模型文件需要手动处理
