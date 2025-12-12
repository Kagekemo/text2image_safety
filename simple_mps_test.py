#!/usr/bin/env python3
"""
简单的LSTM MPS兼容性测试
"""

try:
    import torch
    import torch.nn as nn
    
    print("PyTorch版本:", torch.__version__)
    print("MPS可用:", torch.backends.mps.is_available())
    
    # 测试基本的MPS功能
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用MPS设备")
        
        # 测试简单张量操作
        x = torch.randn(5, 3).to(device)
        y = torch.randn(3, 4).to(device)
        z = torch.mm(x, y)
        print("基本矩阵运算成功:", z.shape)
        
        # 测试LSTM - 这是问题所在
        try:
            lstm = nn.LSTM(10, 20, batch_first=True).to(device)
            input_tensor = torch.randn(5, 3, 10).to(device)
            output, (h, c) = lstm(input_tensor)
            print("LSTM操作成功:", output.shape)
        except Exception as e:
            print("LSTM操作失败:", str(e))
            print("这是已知的MPS限制，我们的兼容层会处理这个问题")
    
    else:
        print("MPS不可用，将使用CPU")
        
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装PyTorch并激活了正确的conda环境")
except Exception as e:
    print(f"测试错误: {e}")
