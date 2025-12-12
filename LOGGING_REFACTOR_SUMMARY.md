# 日志记录模块重构总结

## 概述
本次重构将NSFW检测和日志记录逻辑从 `text2image_pipeline.py` 迁移到独立的 `logging_utils.py` 模块，并且使计算方法与DiffusionDB数据集保持一致。

## 主要变更

### 1. 新建文件: `logging_utils.py`

创建了独立的日志记录模块，包含 `SafetyLogger` 类，负责所有NSFW检测和日志记录功能。

### 2. Prompt NSFW计算逻辑更新

**原来的实现:**
```python
# 使用 Detoxify('unbiased') 模型
# 返回 max(toxicity, sexual_explicit)
```

**新的实现 (与DiffusionDB一致):**
```python
# 使用 Detoxify('multilingual') 模型
# 返回两个独立的值: (toxicity, sexual_explicit)
```

对应 `detect-toxic-prompt.py` 中的逻辑：
```python
result = toxicity_model.predict(cur_prompts)
prompt_toxicity_map[p] = [result["toxicity"][i], result["sexual_explicit"][i]]
```

### 3. Image NSFW计算逻辑实现

**新实现 (与DiffusionDB一致):**

提供了两个方法：

1. **`compute_image_nsfw_from_pil(pil_image)`**: 完整实现
   - 使用 `nsfweffnetv2-b02-3epochs.h5` 模型
   - 检测图像锐度（sharpness），模糊图像（sharpness < 10）直接标记为NSFW (score=2.0)
   - 多分类预测：drawing, hentai, neutral, porn, sexy
   - 使用转换矩阵 `[0.0, 1.0, 0.0, 1.0, 1.0]` 转为二进制分数

2. **`compute_image_nsfw(images_embed)`**: 简化版本
   - 基于CLIP embeddings的快速方法
   - 用于只有embeddings可用的场景

对应 `detect-nsfw-image.py` 中的逻辑：
```python
# 检测锐度
sharpness = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))
if sharpness < 10:
    nsfw_score = 2.0

# 多分类转二进制
trans_mat = np.array([[0.0, 1.0, 0.0, 1.0, 1.0]]).transpose()
nsfw_score_binary = np.dot(nsfw_scores_prob, trans_mat)
```

### 4. CSV日志格式更新

**原来的格式:**
```
prompt, user_name, timestamp, prompt_nsfw, image_nsfw, nsfw_result_bool
```

**新的格式:**
```
prompt, user_name, timestamp, prompt_toxicity, prompt_sexual_explicit, image_nsfw, nsfw_result_bool
```

现在 prompt_nsfw 被分为两列：
- `prompt_toxicity`: 毒性分数
- `prompt_sexual_explicit`: 性暗示分数

## 使用方法

### 在 text2image_pipeline.py 中：

```python
from logging_utils import SafetyLogger

class SDPipeline():
    def __init__(self, device, mode="ti_sd", fix_seed=False):
        # ... 其他初始化 ...
        self.safety_logger = SafetyLogger(device=device)
    
    def __call__(self, text_inputs):
        # ... 生成图像 ...
        
        # 记录日志
        self.safety_logger.log_request(
            prompt_value, 
            images_embed, 
            nsfw_result, 
            pil_image=pil_images[0]
        )
```

### 独立使用 SafetyLogger:

```python
from logging_utils import SafetyLogger
import torch

# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = SafetyLogger(device, csv_path='my_log.csv')

# 计算prompt NSFW
toxicity, sexual = logger.compute_prompt_nsfw("some prompt")

# 从PIL图像计算image NSFW
from PIL import Image
img = Image.open('test.jpg')
nsfw_score = logger.compute_image_nsfw_from_pil(img)

# 记录完整日志
logger.log_request(
    prompt="a cat",
    images_embed=some_embeddings,
    nsfw_result_bool=False,
    pil_image=img
)
```

## 依赖要求

新增的依赖（需要安装）：
- `detoxify`: prompt毒性检测
- `opencv-python` (cv2): 图像锐度计算
- `tensorflow-io`: 图像格式处理（用于WebP等格式）

NSFW图像模型文件：
- 需要下载 `nsfweffnetv2-b02-3epochs.h5` 
- 放置在 `./model/NSFW-cache/` 目录下
- 下载链接: https://github.com/poloclub/diffusiondb

## 优势

1. **代码组织**: 日志逻辑独立，便于维护和测试
2. **一致性**: 与DiffusionDB数据集完全一致的计算方法
3. **可复用**: SafetyLogger可在其他项目中使用
4. **灵活性**: 支持多种图像输入方式（PIL或embeddings）
5. **详细记录**: 分开记录toxicity和sexual_explicit，提供更细粒度的数据

## 注意事项

1. 首次运行会自动下载 Detoxify 模型
2. 如果未找到NSFW图像模型，会降级使用embeddings方法
3. MPS设备（Apple Silicon）使用float32，其他设备使用float16
4. 图像NSFW检测优先使用PIL图像方法以获得最准确的结果
