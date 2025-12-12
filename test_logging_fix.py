"""
测试日志模块修复
验证模型加载和NSFW检测功能
"""

import torch
from logging_utils import SafetyLogger
from PIL import Image
import numpy as np

def test_logger_initialization():
    """测试SafetyLogger初始化"""
    print("=" * 60)
    print("测试 SafetyLogger 初始化...")
    print("=" * 60)
    
    # 检测可用设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("✓ 使用 CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✓ 使用 MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print("✓ 使用 CPU")
    
    logger = SafetyLogger(device, csv_path='test_requests.csv')
    print("✓ SafetyLogger 初始化成功")
    return logger

def test_prompt_nsfw(logger):
    """测试prompt NSFW检测"""
    print("\n" + "=" * 60)
    print("测试 Prompt NSFW 检测...")
    print("=" * 60)
    
    test_prompts = [
        "A beautiful sunset over the ocean",
        "explicit sexual content",
        "A cute cat playing with a ball"
    ]
    
    for prompt in test_prompts:
        try:
            toxicity, sexual = logger.compute_prompt_nsfw(prompt)
            print(f"\nPrompt: '{prompt}'")
            print(f"  - Toxicity: {toxicity:.4f}")
            print(f"  - Sexual Explicit: {sexual:.4f}")
        except Exception as e:
            print(f"\n✗ Error with prompt '{prompt}': {e}")

def test_image_nsfw(logger):
    """测试图像NSFW检测"""
    print("\n" + "=" * 60)
    print("测试 Image NSFW 检测...")
    print("=" * 60)
    
    # 创建一个测试图像 (纯色图像)
    test_img = Image.new('RGB', (256, 256), color=(73, 109, 137))
    
    try:
        nsfw_score = logger.compute_image_nsfw(test_img)
        print(f"\n测试图像 NSFW 分数: {nsfw_score:.4f}")
        print("✓ 图像NSFW检测正常工作")
    except Exception as e:
        print(f"\n✗ 图像NSFW检测错误: {e}")

def test_log_request(logger):
    """测试日志记录功能"""
    print("\n" + "=" * 60)
    print("测试日志记录功能...")
    print("=" * 60)
    
    test_img = Image.new('RGB', (256, 256), color=(100, 150, 200))
    
    try:
        logger.log_request(
            prompt="Test prompt for logging",
            pil_image=test_img,
            nsfw_result_bool=False,
            user_name="test_user"
        )
        print("✓ 日志记录成功")
        print("  查看: test_requests.csv")
    except Exception as e:
        print(f"\n✗ 日志记录错误: {e}")

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("SafetyLogger 功能测试")
    print("=" * 60)
    
    try:
        logger = test_logger_initialization()
        test_prompt_nsfw(logger)
        test_image_nsfw(logger)
        test_log_request(logger)
        
        print("\n" + "=" * 60)
        print("✓ 所有测试完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
