# Mac MPS 支持修复总结

## 问题诊断

原始错误是由于在Mac的MPS（Metal Performance Shaders）设备上运行LSTM时出现的兼容性问题：

1. **LSTM MPS兼容性问题**: MPS对某些RNN操作（特别是LSTM）的支持有限
2. **设备张量不匹配**: 参数张量在MPS设备上，但输入张量被移动到CPU上
3. **API弃用警告**: 使用了过时的`use_auth_token`和`decode_latents`API

## 修复方案

### 1. LSTM MPS兼容层
创建了`MPS_Compatible_LSTM`类，自动处理MPS设备上的LSTM操作：
- 在MPS设备上自动将LSTM操作回退到CPU
- 保持输入输出张量在正确的设备上
- 透明处理设备间的数据传输

### 2. API更新
- 移除了弃用的`use_auth_token=True`参数
- 使用新的VAE解码API替代`decode_latents`方法

### 3. 设备管理优化
- 改进了设备检测和选择逻辑
- 添加了设备特定的错误处理
- 优化了内存管理和缓存清理

## 修复的文件

1. **rl_search.py**
   - 添加了`MPS_Compatible_LSTM`类
   - 修复了设备移动逻辑
   - 改进了错误处理

2. **text2image_pipeline.py** 
   - 移除`use_auth_token`参数
   - 更新VAE解码方法
   - 优化MPS设备处理

3. **device_utils.py**
   - 改进设备检测
   - 添加兼容性函数
   - 增强错误处理

4. **main.py**
   - 添加设备信息显示
   - 集成新的设备管理

## 使用方法

### 快速启动
```bash
# 激活环境
conda activate sneakyprompt

# 测试设备支持
python test_lstm_fix.py

# 运行项目
python main.py --target='sd' --method='rl' --reward_mode='clip' --threshold=0.26 --len_subword=10 --q_limit=60 --safety='ti_sd'
```

### 设备检查
```bash
python device_utils.py
```

## 预期行为

1. **自动设备选择**: 优先使用MPS > CUDA > CPU
2. **透明LSTM处理**: LSTM操作在CPU上执行，其他操作在MPS上
3. **无错误运行**: 不再出现设备不匹配错误
4. **性能提升**: 大部分操作利用Apple Silicon GPU加速

## 注意事项

1. **首次运行较慢**: 需要下载模型文件
2. **内存使用**: MPS和CPU之间的数据传输会增加内存使用
3. **部分操作在CPU**: LSTM相关操作会自动回退到CPU，这是正常的
4. **兼容性**: 支持macOS 12.3+和Apple Silicon芯片

## 故障排除

如果仍有问题：

1. **检查PyTorch版本**: 确保>=2.0.0
2. **重新安装依赖**: 运行`./install_mac_mps.sh`
3. **使用CPU模式**: 
   ```python
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```
4. **查看详细错误**: 运行测试脚本查看具体问题

## 性能对比

- **MPS模式**: 大部分操作GPU加速，LSTM在CPU
- **纯CPU模式**: 所有操作在CPU，较慢但稳定
- **预期性能**: 比纯CPU快30-50%，比CUDA略慢
