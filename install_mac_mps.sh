#!/bin/bash

# Mac MPS 支持的安装脚本
echo "安装支持 Mac MPS 的依赖包..."

# 检查是否为 macOS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "检测到 macOS 系统"
    
    # 安装支持 MPS 的 PyTorch
    echo "安装支持 MPS 的 PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    
    # 安装其他必要依赖
    echo "安装其他依赖..."
    pip install diffusers transformers accelerate safetensors
    pip install pillow torchmetrics
    pip install pandas numpy scipy
    pip install tensorflow tensorflow-hub
    pip install scikit-learn scikit-image
    pip install matplotlib seaborn
    pip install requests tqdm
    
    echo "安装完成！"
    echo "请运行 python device_utils.py 来检查设备支持情况"
    
else
    echo "此脚本专为 macOS 设计。对于其他系统，请使用 environment.yml"
fi
