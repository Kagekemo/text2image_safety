#!/usr/bin/env python3
"""
测试 Mac MPS 支持的脚本
"""

import torch
import warnings
from device_utils import get_optimal_device, print_device_info, clear_cache

def test_mps_basic():
    """基本 MPS 功能测试"""
    print("=== 基本 MPS 功能测试 ===")
    
    device, device_info = get_optimal_device()
    print(f"选择的设备: {device} ({device_info})")
    
    # 测试张量创建和计算
    try:
        # 创建测试张量
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        
        # 矩阵乘法
        z = torch.mm(x, y)
        print(f"矩阵乘法测试通过: {z.shape}")
        
        # 测试梯度计算
        x.requires_grad_(True)
        loss = torch.sum(z * x)
        loss.backward()
        print(f"梯度计算测试通过: {x.grad is not None}")
        
        # 清理缓存
        clear_cache(device)
        print("缓存清理完成")
        
        return True
        
    except Exception as e:
        print(f"基本测试失败: {e}")
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n=== 模型加载测试 ===")
    
    device, _ = get_optimal_device()
    
    try:
        # 创建简单的神经网络
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).to(device)
        
        # 测试前向传播
        x = torch.randn(32, 100).to(device)
        output = model(x)
        print(f"模型前向传播测试通过: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"模型测试失败: {e}")
        return False

def test_diffusion_compatibility():
    """测试 Diffusion 相关依赖"""
    print("\n=== Diffusion 兼容性测试 ===")
    
    try:
        # 测试 transformers
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        print("✓ transformers 库可用")
        
        # 测试 diffusers
        from diffusers import StableDiffusionPipeline
        print("✓ diffusers 库可用")
        
        # 测试 PIL
        from PIL import Image
        print("✓ PIL 库可用")
        
        # 测试 torchvision
        import torchvision.transforms as transforms
        print("✓ torchvision 库可用")
        
        return True
        
    except ImportError as e:
        print(f"依赖库缺失: {e}")
        return False
    except Exception as e:
        print(f"兼容性测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始 Mac MPS 支持测试...\n")
    
    # 打印设备信息
    print_device_info()
    print()
    
    # 检查 MPS 是否可用
    if torch.backends.mps.is_available():
        print("✓ MPS 后端可用")
    else:
        print("✗ MPS 后端不可用")
        print("这可能是因为:")
        print("1. 不是 Apple Silicon Mac")
        print("2. macOS 版本过低 (需要 macOS 12.3+)")
        print("3. PyTorch 版本不支持 MPS")
    
    print()
    
    # 运行测试
    tests = [
        ("基本功能", test_mps_basic),
        ("模型加载", test_model_loading),
        ("依赖兼容性", test_diffusion_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} 测试出现异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n=== 测试结果总结 ===")
    all_passed = True
    for test_name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！Mac MPS 支持正常。")
    else:
        print("\n⚠️ 部分测试失败，可能需要检查环境配置。")
        print("建议运行: ./install_mac_mps.sh 来安装依赖")

if __name__ == "__main__":
    main()
