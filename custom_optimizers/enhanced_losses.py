import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class PixelContrastiveLoss(nn.Module):
    """
    SPCL 像素级对比损失，适配医学图像分割任务
    结合语义原型进行类别对齐，提高跨域适应性能
    """
    def __init__(self, num_classes=2, feature_dim=512, temperature=0.01):
        super(PixelContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.ce_criterion = nn.CrossEntropyLoss(ignore_index=255)
        
    def forward(self, features: torch.Tensor, predictions: torch.Tensor, 
                prototypes: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) 特征图
            predictions: (B, C, H, W) 预测logits  
            prototypes: (C, num_classes) 语义原型
            masks: (B, H, W) 掩码/伪标签
        """
        B, C, H, W = features.size()
        
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        features = features.permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        # 归一化原型
        prototypes = F.normalize(prototypes, p=2, dim=0)
        
        # 计算相似度
        similarity = torch.mm(features, prototypes) / self.temperature
        
        # 展平mask
        masks = masks.view(-1)
        
        # 计算对比损失
        loss = self.ce_criterion(similarity, masks)
        
        return loss


class AdaptivePseudoLabel:
    """
    自适应伪标签生成器，结合SPCL的阈值策略
    专为医学图像分割优化
    """
    def __init__(self, num_classes=2, confidence_threshold=0.95):
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.prob_history = []
        self.label_history = []
        self.thresholds = None
        
    def update_history(self, predictions: torch.Tensor):
        """更新预测历史，用于计算自适应阈值"""
        probs = F.softmax(predictions.detach(), dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        
        self.prob_history.append(max_probs.cpu().numpy().flatten())
        self.label_history.append(pseudo_labels.cpu().numpy().flatten())
        
    def compute_adaptive_thresholds(self, percentile=50):
        """计算每个类别的自适应置信度阈值"""
        if not self.prob_history:
            return [self.confidence_threshold] * self.num_classes
            
        all_probs = np.concatenate(self.prob_history)
        all_labels = np.concatenate(self.label_history)
        
        thresholds = []
        for class_idx in range(self.num_classes):
            class_probs = all_probs[all_labels == class_idx]
            if len(class_probs) > 0:
                threshold = np.percentile(class_probs, percentile)
                threshold = min(threshold, self.confidence_threshold)
            else:
                threshold = self.confidence_threshold
            thresholds.append(threshold)
            
        self.thresholds = thresholds
        return thresholds
        
    def generate_pseudo_labels(self, predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成高质量伪标签"""
        probs = F.softmax(predictions, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1)
        
        if self.thresholds is None:
            self.thresholds = [self.confidence_threshold] * self.num_classes
            
        # 创建置信度mask
        confidence_mask = torch.zeros_like(pseudo_labels, dtype=torch.bool)
        
        for class_idx, threshold in enumerate(self.thresholds):
            class_mask = (pseudo_labels == class_idx) & (max_probs > threshold)
            confidence_mask |= class_mask
            
        return pseudo_labels, confidence_mask


class PrototypeManager:
    """
    语义原型管理器，维护和更新类别原型
    """
    def __init__(self, num_classes=2, feature_dim=512, momentum=0.999):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.prototypes = None
        self.initialized = False
        
    def initialize_prototypes(self, features: torch.Tensor, masks: torch.Tensor):
        """初始化原型"""
        self.prototypes = torch.zeros(self.feature_dim, self.num_classes, 
                                     device=features.device, dtype=features.dtype)
        
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).contiguous().view(-1, C)
        masks_flat = masks.view(-1)
        
        # 计算每个类别的原型
        for class_idx in range(self.num_classes):
            class_mask = (masks_flat == class_idx)
            if class_mask.sum() > 0:
                class_features = features_flat[class_mask]
                prototype = class_features.mean(dim=0)
                self.prototypes[:, class_idx] = F.normalize(prototype, p=2, dim=0)
                
        self.initialized = True
        
    def update_prototypes(self, features: torch.Tensor, masks: torch.Tensor):
        """动量更新原型"""
        if not self.initialized:
            self.initialize_prototypes(features, masks)
            return
            
        # 归一化特征
        features = F.normalize(features, p=2, dim=1)
        B, C, H, W = features.shape
        features_flat = features.permute(0, 2, 3, 1).contiguous().view(-1, C)
        masks_flat = masks.view(-1)
        
        # 更新每个类别的原型
        for class_idx in range(self.num_classes):
            class_mask = (masks_flat == class_idx)
            if class_mask.sum() > 0:
                class_features = features_flat[class_mask]
                new_prototype = class_features.mean(dim=0)
                new_prototype = F.normalize(new_prototype, p=2, dim=0)
                
                # 动量更新
                self.prototypes[:, class_idx] = (
                    self.momentum * self.prototypes[:, class_idx] + 
                    (1 - self.momentum) * new_prototype
                )
                self.prototypes[:, class_idx] = F.normalize(
                    self.prototypes[:, class_idx], p=2, dim=0
                )
    
    def get_prototypes(self) -> Optional[torch.Tensor]:
        """获取当前原型"""
        return self.prototypes if self.initialized else None


class EnhancedMedicalSegLoss:
    """
    增强的医学图像分割损失函数集合
    集成SPCL的先进技术与GraTa的梯度对齐
    """
    def __init__(self, num_classes=2, feature_dim=512):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # SPCL组件
        self.pixel_cl_loss = PixelContrastiveLoss(num_classes, feature_dim)
        self.pseudo_label_generator = AdaptivePseudoLabel(num_classes)
        self.prototype_manager = PrototypeManager(num_classes, feature_dim)
        
        # 传统损失
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def pixel_contrastive_loss(self, features: torch.Tensor, predictions: torch.Tensor, 
                              masks: torch.Tensor) -> torch.Tensor:
        """像素级对比损失"""
        # 更新原型
        self.prototype_manager.update_prototypes(features, masks)
        
        # 获取原型
        prototypes = self.prototype_manager.get_prototypes()
        if prototypes is None:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        # 计算对比损失
        return self.pixel_cl_loss(features, predictions, prototypes, masks)
    
    def enhanced_pseudo_loss(self, features: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """增强的伪标签损失"""
        # 更新历史
        self.pseudo_label_generator.update_history(predictions)
        
        # 计算自适应阈值
        thresholds = self.pseudo_label_generator.compute_adaptive_thresholds()
        
        # 生成伪标签
        pseudo_labels, confidence_mask = self.pseudo_label_generator.generate_pseudo_labels(predictions)
        
        if confidence_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        # 计算原型对比损失
        prototypes = self.prototype_manager.get_prototypes()
        if prototypes is not None:
            return self.pixel_cl_loss(features, predictions, prototypes, pseudo_labels)
        else:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
    
    def multi_scale_consistency_loss(self, pred_original: torch.Tensor, 
                                   pred_augmented: torch.Tensor) -> torch.Tensor:
        """多尺度一致性损失"""
        # 确保尺寸匹配
        if pred_original.shape != pred_augmented.shape:
            pred_augmented = F.interpolate(pred_augmented, size=pred_original.shape[-2:], 
                                         mode='bilinear', align_corners=False)
        
        return self.mse_loss(torch.sigmoid(pred_original), torch.sigmoid(pred_augmented))