import torch
from torch.utils.data import Dataset
import clip
from PIL import Image
import os
import json
import pickle
import numpy as np
import logging
from collections import Counter

class OCLDataset(Dataset):
    def __init__(self, root_dir, split='test', top_k_categories=10):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.root_dir = root_dir
        self.split = split
        self.top_k_categories = top_k_categories
        
        # 初始化 CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        torch.cuda.empty_cache()

        # 加载注释
        pkl_dir = os.path.join(root_dir, "")
        pkl_path = os.path.join(pkl_dir, f"OCL_annot_{split}.pkl")
        
        self.logger.info(f"加载注释文件 {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.annotations = pickle.load(f)
            
        # 加载类别信息
        self.load_class_info(pkl_dir)
        
        # 过滤出前 k 个类别
        self.filter_top_categories()

    def precompute_text_embeddings(self):
        """预计算类别、属性和可操作性的文本嵌入"""
        self.logger.info("预计算文本嵌入...")
        
        with torch.no_grad():
            # 1. 类别文本特征
            categories = list(self.top_categories.keys())
            category_texts = [f"a photo of a {category}" for category in categories]
            category_tokens = clip.tokenize(category_texts).to(self.device)
                # 提取图像特征
            category_features = self.model.encode_text(category_tokens)
            self.text_features = category_features / category_features.norm(dim=-1, keepdim=True)         
            
            # 2. 属性文本特征
            attribute_texts = [f"a {attr} object" for attr in self.attrs]
            attribute_tokens = clip.tokenize(attribute_texts).to(self.device)
            attribute_features = self.model.encode_text(attribute_tokens)
            self.attr_text_features = attribute_features / attribute_features.norm(dim=-1, keepdim=True)
            
            # 3. 可操作性文本特征
            affordance_texts = [f"an object that can {aff}" for aff in self.affs]
            affordance_tokens = clip.tokenize(affordance_texts).to(self.device)
            affordance_features = self.model.encode_text(affordance_tokens)
            self.aff_text_features = affordance_features / affordance_features.norm(dim=-1, keepdim=True)
        
        # 输出嵌入的形状以供验证
        self.logger.info(f"预计算的特征形状:")
        self.logger.info(f"  类别: {self.text_features.shape}")
        self.logger.info(f"  属性: {self.attr_text_features.shape}")
        self.logger.info(f"  可操作性: {self.aff_text_features.shape}")
        
        # 存储类别信息
        self.categories = categories
        
    def load_class_info(self, pkl_dir):
        """加载类别信息"""
        def load_class_json(name):
            path = os.path.join(pkl_dir, f"OCL_class_{name}.json")
            with open(path, 'r') as f:
                return json.load(f)
                
        self.attrs = load_class_json("attribute")
        self.objs = load_class_json("object")
        self.affs = load_class_json("affordance")
        self.obj2id = {obj: idx for idx, obj in enumerate(self.objs)}
        
        matrix_path = os.path.join(pkl_dir, 'category_aff_matrix.json')
        with open(matrix_path, 'r') as f:
            aff_matrix = json.load(f)
            self.aff_matrix = np.array(aff_matrix["aff_matrix"])

    def filter_top_categories(self):
        """基于样本数量过滤出前 k 个类别"""
        # 统计每个类别的样本数量
        category_counts = {}
        for ann in self.annotations:
            for obj in ann['objects']:
                category = obj['obj']
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # 获取前 k 个类别
        self.top_categories = dict(Counter(category_counts).most_common(self.top_k_categories))
        self.logger.info(f"选择了前 {len(self.top_categories)} 个类别")
        
        # 过滤注释
        self.filtered_annotations = []
        for ann in self.annotations:
            filtered_objects = [obj for obj in ann['objects'] if obj['obj'] in self.top_categories]
            if filtered_objects:
                ann = ann.copy()
                ann['objects'] = filtered_objects
                self.filtered_annotations.append(ann)
        self.logger.info(f"过滤后的数据集包含 {len(self.filtered_annotations)} 张图片")

        # 创建类别到索引的映射
        self.filtered_obj2id = {obj: idx for idx, obj in enumerate(self.top_categories.keys())}
        
        # 过滤后，预计算文本嵌入
        self.precompute_text_embeddings()

    def __len__(self):
        return len(self.filtered_annotations)
        
    def __getitem__(self, idx):
        ann = self.filtered_annotations[idx]
        
        # 加载和预处理图片
        img_path = os.path.join(self.root_dir, "", ann["name"])
        image = Image.open(img_path).convert('RGB')
        
        # 处理大尺寸图片
        if max(image.size) > 1800:
            w, h = image.size
            image = image.resize((w//2, h//2))
                
        # 应用 CLIP 的预处理
        image = self.preprocess(image)
        
        # 获取第一个对象的类别
        obj = ann['objects'][0]
        category = obj['obj']
        
        # 获取属性和可操作性标签
        attr = torch.zeros(len(self.attrs))
        attr[obj['attr']] = 1
        
        aff = torch.zeros(self.aff_matrix.shape[1])
        aff[obj['aff']] = 1
        
        return {
            'image': image,
            'category': category,
            'category_id': self.filtered_obj2id[category],
            'attributes': attr,
            'affordances': aff
        }

"""
假设输入数据如下：
    image = PIL.Image.open('cat.jpg')  # 假设是一张 2000x1500 的图片
    ann = {
        'objects': [
            {'obj': 'cat', 'attr': 2, 'aff': 1}
        ]
    }
    self.attrs = ['small', 'medium', 'fluffy', 'large']
    self.aff_matrix = torch.zeros(4, 4)  # 假设有 4 种可操作性
    self.filtered_obj2id = {'cat': 0, 'dog': 1}

执行后的返回结果：
    'image': torch.Size([3, 224, 224]),  # 经过 CLIP 预处理后的图像张量
    'category': 'cat',
    'category_id': 0,
    'attributes': tensor([0., 0., 1., 0.]),  # 'fluffy' 位置设为 1
    'affordances': tensor([0., 1., 0., 0.])  # 操作性标签为 1
"""