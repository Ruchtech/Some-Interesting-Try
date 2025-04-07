import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 设置CUDA内存分配的最大切分大小为128MB
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                         # 指定使用GPU 0

import torch
import logging
import numpy as np
from torch.utils.data import DataLoader, Subset
from dataset import OCLDataset
from model import OCLModel

def setup_logging():
    """
    设置日志记录器。
    配置日志输出格式，显示时间、日志级别和消息。
    """
    logging.basicConfig(
        level=logging.INFO,                                             # 设置日志级别为INFO，显示信息及以上级别的日志
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 输出格式：时间 - 模块名 - 日志级别 - 消息
        handlers=[logging.StreamHandler()]                              # 输出到标准输出（控制台）
    )
    return logging.getLogger('DemoProgram')                             # 获取日志记录器对象

def create_small_loader(dataset_path, num_samples=100):
    """
    创建一个小型数据加载器，只加载数据集的前num_samples个样本。
    
    参数：
    dataset_path (str): 数据集路径
    num_samples (int): 加载的样本数量
    
    返回：
    loader (DataLoader): 小型数据加载器
    """
    dataset = OCLDataset(             # 加载OCLDataset数据集
        root_dir=dataset_path,
        split='train',                # 使用训练集
        top_k_categories=10           # 设置类别数为前10类
    )
    
    # 创建一个子集，只使用前num_samples个样本
    subset_indices = list(range(min(num_samples, len(dataset))))  # 生成从0到num_samples-1的索引
    subset = Subset(dataset, subset_indices)                      # 创建一个数据集子集
    
    # 创建数据加载器
    loader = DataLoader(
        subset,
        batch_size=4,                 # 批量大小设置为4，便于调试
        shuffle=False,                # 不打乱数据
        num_workers=0                 # 使用单进程，方便调试
    )
    return loader                     # 返回数据加载器

def demo_evaluation(model, loader, logger):
    """
    演示评估模型，重点是分类、属性和效能预测。
    
    参数：
    model (OCLModel): OCL模型实例
    loader (DataLoader): 数据加载器
    logger (logging.Logger): 日志记录器
    """
    with torch.no_grad():                             # 在评估过程中，不计算梯度
        batch = next(iter(loader))                    # 获取数据加载器中的第一个批次数据
        image_features = model.get_embeddings(batch)  # 获取图像特征
        
        # 获取文本特征和类别名称列表  
        cat_sim = model.compute_similarity(image_features, loader.dataset.dataset.text_features.to(model.device))                                             # 计算类别相似度
        attr_sim = model.compute_similarity(image_features, loader.dataset.dataset.attr_text_features.to(model.device))   # 计算属性相似度
        aff_sim = model.compute_similarity(image_features, loader.dataset.dataset.aff_text_features.to(model.device))     # 计算效能相似度

        # 计算类别预测准确性
        cat_accuracy = model.compute_accuracy(cat_sim, batch['category_id'].to(model.device))        
        # 获取属性和效能的标签
        attr_labels = batch['attributes'].to(model.device)
        aff_labels = batch['affordances'].to(model.device)
        
        attr_map = model.compute_map(attr_sim, attr_labels)   # 计算属性的平均准确率
        aff_map = model.compute_map(aff_sim, aff_labels)      # 计算效能的平均准确率
        
        # 输出总体指标
        logger.info("\n=== Overall Metrics ===")
        logger.info(f"物体和属性标签间_准确性: {cat_accuracy:.4f}")  # 类别准确性
        logger.info(f"属性_平均准确率: {attr_map:.4f}")             # 属性平均准确性
        logger.info(f"效能_平均准确率: {aff_map:.4f}")              # 效能平均准确性

def main():
    """
    主函数，设置日志、初始化数据加载器和模型，并运行评估。
    """
    logger = setup_logging()                                                   # 设置日志记录器
    logger.info("Starting demo evaluation...")                                 # 记录开始评估的信息
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 根据是否有GPU选择设备
        logger.info(f"Using device: {device}")                                 # 输出使用的设备信息
        
        dataset_path = './data/resources'                                      # 数据集路径
        
        # 创建一个小数据集的加载器
        loader = create_small_loader(dataset_path)
        logger.info(f"Created small loader with {len(loader)} batches")        # 输出加载器批次数量
        
        # 初始化OCL模型
        model = OCLModel()
        
        # 运行demo评估
        demo_evaluation(model, loader, logger)
        
    except Exception as e:
        # 捕获异常并记录错误
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        torch.cuda.empty_cache()                                               # 清空CUDA缓存

if __name__ == '__main__':
    # 启动主函数
    main()
