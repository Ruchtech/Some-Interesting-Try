import os
import torch
import torch.nn.functional as F
import clip
import logging
import numpy as np
from sklearn.metrics import average_precision_score
from PIL import Image

class OCLModel:
    def __init__(self, model_name="ViT-B/32"):
        # 初始化日志记录器
        self.logger = logging.getLogger(self.__class__.__name__)
        # 判断是否有可用的GPU，否则使用CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 输出模型初始化信息
        self.logger.info(f'Initializing CLIP model {model_name} on {self.device}')
        
        try:
            # 加载CLIP模型和预处理函数
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval()          # 设置模型为评估模式
            self._freeze_parameters()  # 冻结模型参数
        except Exception as e:
            # 如果加载模型失败，记录错误信息并抛出异常
            self.logger.error(f'Failed to initialize model: {e}')
            raise
            
    def _freeze_parameters(self):
        """冻结模型参数"""
        # 冻结所有参数，使其不可训练
        for param in self.model.parameters():
            param.requires_grad = False
            
    def get_embeddings(self, batch):
        """获取图像的嵌入特征"""
        try:
            images = batch['image'].to(self.device)                                          # 将图像数据转移到设备（GPU或CPU）
            with torch.no_grad():                                                            # 不计算梯度
                image_features = self.model.encode_image(images)                             # 获取图像特征
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化图像特征
            return image_features
        except Exception as e:
            # 如果前向传播失败，记录错误信息并抛出异常
            self.logger.error(f'Forward pass failed: {e}')
            raise
            
    def compute_similarity(self, image_features, text_features):
        """计算图像特征与文本特征之间的余弦相似性"""
        # 归一化图像特征和文本特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算图像特征与文本特征之间的余弦相似性
        similarity = torch.mm(image_features, text_features.t())                             # 使用矩阵乘法计算相似性
        return similarity


def main():
    # 创建OCLModel实例
    model = OCLModel()

    # 加载并处理图片
    image_path = "dog.png"
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return

    # 预处理图像
    image_input = model.preprocess(image).unsqueeze(0).to(model.device)

    # 获取图像特征
    image_features = model.get_embeddings({'image': image_input})

    # 创建文本 "a dog" 的特征
    text_input = torch.cat([clip.tokenize(["dog"])], dim=0).to(model.device)
    with torch.no_grad():
        text_features = model.model.encode_text(text_input)
    
    # 计算图像与文本的相似度
    similarity = model.compute_similarity(image_features, text_features)
    
    # 输出文本与图像的相关性
    print(f"Similarity between 'a dog' and the image: {similarity.item()}")

if __name__ == "__main__":
    main()
