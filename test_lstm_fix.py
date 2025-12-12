#!/usr/bin/env python3
"""
测试LSTM MPS兼容性的修复
"""

import sys
import os
sys.path.append('/Users/artdeco/Desktop/Deepfake/text2image_safety')

try:
    import torch
    import torch.nn as nn
    from rl_search import robot
    from device_utils import get_optimal_device
    
    print("=== 测试LSTM MPS兼容性修复 ===")
    
    # 获取设备
    device, device_info = get_optimal_device()
    print(f"使用设备: {device} ({device_info})")
    
    # 测试MPS兼容的LSTM
    print("\n测试MPS兼容LSTM...")
    try:
        lstm_layer = robot.MPS_Compatible_LSTM(input_size=10, hidden_size=20, batch_first=True)
        lstm_layer = lstm_layer.to(device)
        
        # 创建测试输入
        batch_size = 2
        seq_len = 5
        input_size = 10
        
        x = torch.randn(batch_size, seq_len, input_size).to(device)
        print(f"输入张量形状: {x.shape}, 设备: {x.device}")
        
        # 前向传播
        output, (h, c) = lstm_layer(x)
        print(f"输出张量形状: {output.shape}, 设备: {output.device}")
        print(f"隐藏状态形状: {h.shape}, 设备: {h.device}")
        print("✓ LSTM测试成功!")
        
        # 测试带隐藏状态的连续调用
        print("\n测试连续LSTM调用...")
        x2 = torch.randn(batch_size, seq_len, input_size).to(device)
        output2, (h2, c2) = lstm_layer(x2, (h, c))
        print(f"第二次输出形状: {output2.shape}, 设备: {output2.device}")
        print("✓ 连续LSTM调用成功!")
        
    except Exception as e:
        print(f"✗ LSTM测试失败: {e}")
        
    # 测试完整的p_pi网络
    print("\n测试完整的p_pi网络...")
    try:
        space = [100, 100, 100]  # 示例搜索空间
        policy_net = robot.p_pi(space, embedding_size=30, stable=True, v_theta=True)
        policy_net = policy_net.to(device)
        
        # 测试前向传播
        state = torch.zeros((1, 1)).long().to(device)
        prob, value = policy_net(state)
        print(f"策略输出形状: {prob.shape}, 设备: {prob.device}")
        print(f"价值输出形状: {value.shape}, 设备: {value.device}")
        print("✓ 完整网络测试成功!")
        
    except Exception as e:
        print(f"✗ 完整网络测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n=== 测试完成 ===")
    
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保在正确的conda环境中运行此脚本")
except Exception as e:
    print(f"测试中出现未预期的错误: {e}")
    import traceback
    traceback.print_exc()
