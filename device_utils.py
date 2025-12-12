"""
设备工具模块 - 支持Mac MPS、CUDA和CPU
"""
import torch
import warnings

def get_optimal_device():
    """
    获取最优设备，优先级：MPS > CUDA > CPU
    返回设备对象和设备信息
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_info = "Apple Silicon GPU (MPS)"
        # MPS 的一些限制和建议
        warnings.warn(
            "使用 MPS 设备。注意：LSTM等某些操作会自动回退到 CPU 以确保兼容性。",
            UserWarning
        )
        return device, device_info
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_info = f"CUDA GPU: {torch.cuda.get_device_name()}"
        return device, device_info
    else:
        device = torch.device("cpu")
        device_info = "CPU"
        return device, device_info

def clear_cache(device):
    """
    根据设备类型清理缓存
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    # CPU 不需要特殊的缓存清理

def get_recommended_dtype(device):
    """
    根据设备获取推荐的数据类型
    """
    if device.type == "mps":
        # MPS 对 float16 支持有限，推荐使用 float32
        return torch.float32
    elif device.type == "cuda":
        # CUDA 通常支持 float16 以节省内存
        return torch.float16
    else:
        # CPU 使用 float32
        return torch.float32

def setup_generator(device, seed=None):
    """
    根据设备类型设置随机数生成器
    """
    if seed is None:
        return None
    
    if device.type == "mps":
        # MPS 不支持 Generator，使用全局种子
        torch.manual_seed(seed)
        return None
    elif device.type == "cuda":
        generator = torch.Generator(device='cuda')
        generator.manual_seed(seed)
        return generator
    else:
        generator = torch.Generator(device='cpu')
        generator.manual_seed(seed)
        return generator

def safe_tensor_operation(func, *args, **kwargs):
    """
    安全地执行张量操作，在MPS出错时自动回退到CPU
    """
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if "mps" in str(e).lower():
            print(f"MPS操作失败，回退到CPU: {e}")
            # 将参数移动到CPU
            cpu_args = []
            for arg in args:
                if hasattr(arg, 'cpu'):
                    cpu_args.append(arg.cpu())
                else:
                    cpu_args.append(arg)
            
            cpu_kwargs = {}
            for k, v in kwargs.items():
                if hasattr(v, 'cpu'):
                    cpu_kwargs[k] = v.cpu()
                else:
                    cpu_kwargs[k] = v
            
            # 在CPU上执行
            result = func(*cpu_args, **cpu_kwargs)
            
            # 如果原始参数在MPS上，将结果移回MPS
            if args and hasattr(args[0], 'device') and args[0].device.type == 'mps':
                if hasattr(result, 'to'):
                    result = result.to(args[0].device)
            
            return result
        else:
            raise e

def get_device_memory_info(device):
    """
    获取设备内存信息
    """
    if device.type == "cuda":
        return {
            "total": torch.cuda.get_device_properties(device).total_memory,
            "allocated": torch.cuda.memory_allocated(device),
            "cached": torch.cuda.memory_reserved(device)
        }
    elif device.type == "mps":
        # MPS 目前没有直接的内存查询API
        return {"info": "MPS memory info not available"}
    else:
        return {"info": "CPU memory depends on system RAM"}

def print_device_info():
    """
    打印设备信息
    """
    device, device_info = get_optimal_device()
    print(f"使用设备: {device_info}")
    print(f"设备类型: {device.type}")
    
    if device.type == "cuda":
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
    elif device.type == "mps":
        print("MPS 后端可用")
        print("推荐数据类型: float32")
    
    print(f"PyTorch 版本: {torch.__version__}")
    return device, device_info

if __name__ == "__main__":
    print_device_info()
