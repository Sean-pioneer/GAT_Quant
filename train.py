"""
训练脚本
Training Script

实现完整的训练循环，包括:
- 数据加载
- 模型训练
- 验证评估
- 模型保存
- 日志记录
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import os

from config import Config, get_config
from models import MarketSentimentGAT
from utils import (
    set_seed,
    get_device,
    get_loss_function,
    get_scheduler,
    compute_metrics,
    Logger,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
)


class Trainer:
    """
    训练器类
    
    封装完整的训练流程
    """
    
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
    ):
        """
        初始化训练器
        
        Args:
            config: 配置对象
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器（可选）
        """
        self.config = config
        self.device = get_device(config.training.device)
        
        # 模型
        self.model = model.to(self.device)
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # 学习率调度器
        self.scheduler = get_scheduler(
            self.optimizer,
            scheduler_type=config.training.scheduler_type,
            num_epochs=config.training.num_epochs,
            warmup_epochs=config.training.warmup_epochs,
        )
        
        # 损失函数
        self.criterion = get_loss_function(config.training.loss_type)
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode='min',
        )
        
        # 日志
        self.logger = Logger(log_dir=config.log_dir)
        
        # 最佳模型路径
        self.best_model_path = os.path.join(config.output_dir, 'best_model.pt')

    def _unpack_batch(self, batch):
        graph, targets, target_mask = batch
        return graph, targets, target_mask

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_mask is None:
            return self.criterion(predictions, targets)

        valid = target_mask.to(dtype=torch.bool)
        if valid.sum().item() == 0:
            return predictions.sum() * 0.0
        return self.criterion(predictions[valid], targets[valid])
    
    def train_epoch(self) -> float:
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc="Training",
            leave=False,
        )
        
        for batch in progress_bar:
            graph, targets, target_mask = self._unpack_batch(batch)

            targets = targets.to(self.device)
            if target_mask is not None:
                target_mask = target_mask.to(self.device)
            graph = graph.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(graph)
            
            # 计算损失
            loss = self._compute_loss(predictions, targets, target_mask)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / max(num_batches, 1)
    
    @torch.no_grad()
    def evaluate(
        self, 
        data_loader: DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        """
        评估模型
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            平均损失和评估指标
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        for batch in data_loader:
            graph, targets, target_mask = self._unpack_batch(batch)

            targets = targets.to(self.device)
            if target_mask is not None:
                target_mask = target_mask.to(self.device)
            graph = graph.to(self.device)

            predictions = self.model(graph)
            
            # 计算损失
            loss = self._compute_loss(predictions, targets, target_mask)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 收集预测和目标
            pred_np = predictions.cpu().numpy()
            target_np = targets.cpu().numpy()
            if target_mask is not None:
                mask_np = target_mask.cpu().numpy().astype(bool)
                target_np = np.where(mask_np, target_np, np.nan)
            all_predictions.append(pred_np)
            all_targets.append(target_np)
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 计算指标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = compute_metrics(all_predictions, all_targets)
        
        return avg_loss, metrics
    
    def train(self) -> Dict:
        """
        完整训练流程
        
        Returns:
            训练历史
        """
        self.logger.log(f"开始训练，使用设备: {self.device}")
        self.logger.log(f"模型参数量: {count_parameters(self.model):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config.training.num_epochs + 1):
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, val_metrics = self.evaluate(self.val_loader)
            
            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 记录日志
            self.logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=val_metrics,
                lr=current_lr,
            )
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    self.best_model_path,
                )
                self.logger.log(f"保存最佳模型 (val_loss: {val_loss:.4f})")
            
            # 早停检查
            if self.early_stopping(val_loss):
                self.logger.log(f"早停触发，在epoch {epoch}")
                break
        
        # 测试评估
        if self.test_loader is not None:
            if os.path.exists(self.best_model_path):
                self.model, self.optimizer, best_epoch, best_loss = load_checkpoint(
                    self.model,
                    self.optimizer,
                    self.best_model_path,
                    self.device,
                )
                self.logger.log(
                    f"加载最佳模型用于测试 (epoch: {best_epoch}, val_loss: {best_loss:.4f})"
                )
            test_loss, test_metrics = self.evaluate(self.test_loader)
            self.logger.log(f"\n测试集评估:")
            self.logger.log(f"  Loss: {test_loss:.4f}")
            for k, v in test_metrics.items():
                self.logger.log(f"  {k}: {v:.4f}")
        
        return self.logger.history


