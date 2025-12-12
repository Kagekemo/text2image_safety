"""
日志记录工具模块
用于记录文本到图像生成请求的NSFW检测结果
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import cv2
from datetime import datetime
from detoxify import Detoxify

try:
    import tf_keras  # For TensorFlow 2.16+ / Keras 3 compatibility
except Exception:
    tf_keras = None


class SafetyLogger:
    """
    安全性日志记录器
    遵循DiffusionDB数据集中的NSFW计算逻辑
    """
    
    def __init__(self, device, csv_path='pipeline_requests.csv'):
        """
        初始化日志记录器
        
        Args:
            device: torch设备（cuda/cpu/mps）
            csv_path: CSV日志文件路径
        """
        self.device = device
        self.csv_path = csv_path
        self._prompt_classifier = None  # Detoxify模型
        self._image_classifier = None   # NSFW图像分类器
        
    def _init_prompt_classifier(self):
        """初始化prompt毒性分类器（multilingual模型）"""
        if self._prompt_classifier is None:
            # 使用multilingual模型与DiffusionDB保持一致
            self._prompt_classifier = Detoxify('multilingual', device=self.device.type)
    
    def _init_image_classifier(self):
        """初始化图像NSFW分类器"""
        if self._image_classifier is None:
            # 设置TensorFlow使用CPU或GPU
            if self.device.type == 'cuda':
                # 使用GPU
                gpus = tf.config.experimental.list_physical_devices("GPU")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # 加载NSFW模型
            cache_folder = "./model/NSFW-cache"
            model_path = os.path.join(cache_folder, "nsfweffnetv2-b02-3epochs.h5")
            
            if not os.path.exists(model_path):
                print(f"[警告] NSFW图像模型未找到: {model_path}")
                print("请从https://github.com/LAION-AI/LAION-SAFETY下载nsfweffnetv2-b02-3epochs.h5模型")
                return None
            
            try:
                # NOTE:
                # 该 LAION NSFW .h5 模型内部包含 TFHub 的 KerasLayer。
                # 在 TF 2.16+ (tf.keras=Keras 3) 环境下，tf.keras.models.load_model
                # 可能会出现 “Only instances of `keras.Layer`…” 的类型不匹配。
                # 这时应使用 legacy 的 tf_keras 来反序列化该模型。
                use_tf_keras = False
                try:
                    use_tf_keras = (
                        tf_keras is not None
                        and str(getattr(tf.keras, "__version__", "")).startswith("3")
                    )
                except Exception:
                    use_tf_keras = tf_keras is not None

                loader = tf_keras.models.load_model if use_tf_keras else tf.keras.models.load_model

                self._image_classifier = loader(
                    model_path,
                    custom_objects={"KerasLayer": hub.KerasLayer},
                    compile=False,
                )
                print(f"[成功] NSFW图像模型加载成功")
            except Exception as e:
                print(f"[错误] 加载NSFW图像模型失败: {e}")
                if "Only instances of `keras.Layer`" in str(e) or "keras.Layer" in str(e):
                    print(
                        "[提示] 你当前可能在 TF 2.16+/Keras 3 环境中运行。"
                        "该 .h5 模型需要 legacy Keras 反序列化：\n"
                        "- 建议安装/确认 tf-keras：pip install tf-keras\n"
                        "- 或设置环境变量 TF_USE_LEGACY_KERAS=1 后再启动程序\n"
                        "本项目已尝试自动使用 tf_keras；若仍失败，请确认 tf_keras 已可 import。"
                    )
                return None
    
    def compute_prompt_nsfw(self, prompt):
        """
        计算prompt的NSFW分数（遵循DiffusionDB逻辑）
        
        Args:
            prompt: 输入的文本提示词
            
        Returns:
            tuple: (toxicity_score, sexual_explicit_score) 或 (0.0, 0.0) 如果失败
        """
        self._init_prompt_classifier()
        try:
            result = self._prompt_classifier.predict([prompt])
            # 返回toxicity和sexual_explicit两个值
            toxicity = float(result['toxicity'][0])
            sexual_explicit = float(result['sexual_explicit'][0])
            return toxicity, sexual_explicit
        except Exception as e:
            print(f"[prompt nsfw warning] {e}")
            return 0.0, 0.0
    
    def compute_image_nsfw(self, pil_image):
        """
        从PIL图像计算NSFW分数
        
        Args:
            pil_image: PIL图像对象
            
        Returns:
            float: NSFW分数
        """
        self._init_image_classifier()
        if self._image_classifier is None:
            return 0.0
        
        try:
            # 转换PIL图像为numpy数组
            img_array = np.array(pil_image)
            
            # 计算锐度（sharpness）
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            sharpness = np.max(cv2.convertScaleAbs(cv2.Laplacian(gray, 3)))
            
            # 如果图像模糊，直接返回高NSFW分数
            if sharpness < 10:
                return 2.0
            
            # 预处理图像
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            
            # Resize到260x260
            resized_img = tf.image.resize(img_tensor, [260, 260], antialias=True)
            
            # 归一化到0-1
            normalizer = tf.keras.layers.Rescaling(1.0 / 255)
            normalized_img = normalizer(resized_img)
            
            # 添加batch维度
            batched_img = tf.expand_dims(normalized_img, 0)
            
            # 预测
            with tf.device('/cpu:0' if self.device.type != 'cuda' else '/device:GPU:0'):
                nsfw_scores = self._image_classifier.predict(batched_img, verbose=0)
                nsfw_scores_prob = tf.nn.softmax(nsfw_scores).numpy()
                
                # 转换multi-class概率为二进制分数
                # Columns: drawing, hentai, neutral, porn, sexy
                trans_mat = np.array([[0.0, 1.0, 0.0, 1.0, 1.0]]).transpose()
                nsfw_score_binary = float(np.dot(nsfw_scores_prob, trans_mat)[0][0])
                
                return nsfw_score_binary
                
        except Exception as e:
            print(f"[image nsfw from PIL warning] {e}")
            return 0.0
    
    def log_request(self, prompt, pil_image, nsfw_result_bool, user_name="admin"):
        """
        记录一次生成请求
        
        Args:
            prompt: 输入的文本提示词
            pil_image: PIL图像（如果可用，用于更准确的图像NSFW计算）
            nsfw_result_bool: NSFW检测结果（布尔值）
            user_name: 用户名
        """
        file_exists = os.path.isfile(self.csv_path)
        timestamp = datetime.now().isoformat()
        is_nsfw = 1 if nsfw_result_bool else 0
        
        # 计算prompt的NSFW分数（返回两个值）
        prompt_toxicity, prompt_sexual = self.compute_prompt_nsfw(prompt)
        
        # 计算图像的NSFW分数
        image_nsfw = self.compute_image_nsfw(pil_image)
        
        # 准备CSV行数据
        row = [
            prompt, 
            user_name, 
            timestamp, 
            prompt_toxicity,      # toxicity分数
            prompt_sexual,        # sexual_explicit分数
            image_nsfw,          # 图像NSFW分数
            is_nsfw      # 最终NSFW判定结果
        ]
        
        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "prompt", 
                        "user_name", 
                        "timestamp", 
                        "prompt_toxicity",
                        "prompt_sexual_explicit", 
                        "image_nsfw", 
                        "nsfw_result_bool"
                    ])
                writer.writerow(row)
        except Exception as e:
            print(f"[csv write warning] {e}")
