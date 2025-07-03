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

### 方法3：使用外部存储

您也可以将模型文件存储在外部服务上（如Google Drive、AWS S3等），并在README中提供下载链接。

## 当前状态

- ✅ 代码已成功推送到GitHub
- ✅ .gitignore已配置，排除大文件
- ⚠️ 模型文件需要手动处理

## 项目结构

```
WatermarkRemover-AI/
├── models/                    # 模型文件目录（需要手动创建）
│   └── epoch=071-valid_iou=0.7267.ckpt  # 模型文件（需要手动添加）
├── scripts/                   # 脚本文件
├── test/                      # 测试文件
├── archive/                   # 归档文件
├── requirements.txt           # Python依赖
├── setup.sh                   # 安装脚本
└── README.md                  # 项目说明
```

## 下一步

1. 按照上述方法之一处理模型文件
2. 运行 `python setup.py` 或 `bash setup.sh` 安装依赖
3. 按照README.md中的说明运行项目 