#!/bin/bash

echo "启动 SneakyPrompt with MPS 支持..."

# 检查是否在正确的环境中
if [[ "$CONDA_DEFAULT_ENV" != "sneakyprompt" ]]; then
    echo "正在激活 sneakyprompt 环境..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate sneakyprompt
fi

# 检查设备支持
echo "检查设备支持..."
python3 simple_mps_test.py

echo ""
echo "如果上述测试通过，您可以运行以下命令启动项目："
echo "python main.py --target='sd' --method='rl' --reward_mode='clip' --threshold=0.26 --len_subword=10 --q_limit=60 --safety='ti_sd'"
