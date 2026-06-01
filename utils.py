"""
工具函数模块
Utility Functions Module

包含训练辅助函数、评估指标、日志记录等
"""

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import Dict, List, Optional, Tuple
import random
import os
from datetime import datetime


def set_seed(seed: int = 42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """
    统计模型可训练参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    获取计算设备
    
    Args:
        device_str: 设备字符串 ('cuda', 'mps', 'cpu', None自动选择)
        
    Returns:
        torch.device对象
    """
    if device_str is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


# ============================================
# 损失函数
# ============================================

class MSELoss(nn.Module):
    """标准MSE损失"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.mse(predictions, targets)


class QuasiLikelihoodLoss(nn.Module):
    """
    准似然损失函数 (Quasi-Likelihood Loss)
    
    相比MSE，对异方差性(Heteroskedasticity)处理更好
    特别适合波动率预测
    
    公式:
    L = log(σ²) + (y - ŷ)² / σ²
    
    其中σ²是预测的方差
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        variance: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算准似然损失
        
        Args:
            predictions: 预测值
            targets: 目标值
            variance: 预测的方差（如果没有，使用预测值的平方）
            
        Returns:
            损失值
        """
        if variance is None:
            # 使用简化版本：使用残差的绝对值作为近似方差
            residuals = targets - predictions
            variance = torch.abs(residuals).detach() + self.eps
        
        loss = torch.log(variance + self.eps) + (targets - predictions) ** 2 / (variance + self.eps)
        return loss.mean()


class ICLoss(nn.Module):
    """
    信息系数(IC)损失
    
    最大化预测值与真实收益率的秩相关性
    适用于因子模型
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        计算IC损失（负的秩相关性）
        
        Args:
            predictions: 预测值
            targets: 目标值
            
        Returns:
            负的IC值（用于最小化）
        """
        # 转换为秩
        pred_ranks = self._rank(predictions)
        target_ranks = self._rank(targets)
        
        # 计算相关性
        pred_mean = pred_ranks.mean()
        target_mean = target_ranks.mean()
        
        pred_centered = pred_ranks - pred_mean
        target_centered = target_ranks - target_mean
        
        correlation = (pred_centered * target_centered).sum() / \
                      (torch.sqrt((pred_centered ** 2).sum()) * 
                       torch.sqrt((target_centered ** 2).sum()) + 1e-8)
        
        return -correlation  # 负号用于最小化
    
    @staticmethod
    def _rank(x: Tensor) -> Tensor:
        """计算排名"""
        sorted_indices = torch.argsort(x)
        ranks = torch.zeros_like(x)
        ranks[sorted_indices] = torch.arange(len(x), dtype=x.dtype, device=x.device)
        return ranks


def get_loss_function(loss_type: str = "mse") -> nn.Module:
    """
    获取损失函数
    
    Args:
        loss_type: 损失函数类型 ('mse', 'quasi_likelihood', 'ic')
        
    Returns:
        损失函数模块
    """
    loss_functions = {
        'mse': MSELoss(),
        'quasi_likelihood': QuasiLikelihoodLoss(),
        'ic': ICLoss(),
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_type]


# ============================================
# 评估指标
# ============================================

def compute_metrics(
    predictions: np.ndarray, 
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        predictions: 预测值
        targets: 目标值
        
    Returns:
        指标字典
    """
    metrics: Dict[str, float] = {}

    # 金融选股更关心每个交易日横截面上的排序能力。
    if predictions.ndim == 2 and targets.ndim == 2 and predictions.shape == targets.shape:
        daily_ic_values = []
        daily_rank_ic_values = []
        from scipy.stats import spearmanr

        for pred_row, target_row in zip(predictions, targets):
            valid = np.isfinite(pred_row) & np.isfinite(target_row)
            if valid.sum() < 3:
                continue
            pred_valid = pred_row[valid]
            target_valid = target_row[valid]
            if np.std(pred_valid) > 1e-8 and np.std(target_valid) > 1e-8:
                daily_ic_values.append(np.corrcoef(pred_valid, target_valid)[0, 1])
                rank_ic, _ = spearmanr(pred_valid, target_valid)
                if not np.isnan(rank_ic):
                    daily_rank_ic_values.append(rank_ic)

        if daily_ic_values:
            daily_ic = np.asarray(daily_ic_values, dtype=np.float64)
            metrics["daily_ic"] = float(daily_ic.mean())
            metrics["daily_ic_std"] = float(daily_ic.std(ddof=0))
            metrics["daily_ic_ir"] = float(
                daily_ic.mean() / (daily_ic.std(ddof=0) + 1e-8)
            )
        else:
            metrics["daily_ic"] = 0.0
            metrics["daily_ic_std"] = 0.0
            metrics["daily_ic_ir"] = 0.0

        if daily_rank_ic_values:
            daily_rank_ic = np.asarray(daily_rank_ic_values, dtype=np.float64)
            metrics["daily_rank_ic"] = float(daily_rank_ic.mean())
        else:
            metrics["daily_rank_ic"] = 0.0

    # 展平后给出整体回归误差和全局相关性。
    predictions = predictions.flatten()
    targets = targets.flatten()
    valid = np.isfinite(predictions) & np.isfinite(targets)
    predictions = predictions[valid]
    targets = targets[valid]
    if predictions.size == 0:
        metrics.update({
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0,
            'ic': 0.0,
            'rank_ic': 0.0,
            'direction_accuracy': 0.0,
        })
        return metrics
    
    # MSE
    mse = np.mean((predictions - targets) ** 2)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(predictions - targets))
    
    # IC (Information Coefficient)
    ic = np.corrcoef(predictions, targets)[0, 1]
    if np.isnan(ic):
        ic = 0.0
    
    # 秩IC (Rank IC)
    from scipy.stats import spearmanr
    rank_ic, _ = spearmanr(predictions, targets)
    if np.isnan(rank_ic):
        rank_ic = 0.0
    
    # 方向准确率
    direction_accuracy = np.mean(
        (predictions > 0) == (targets > 0)
    )
    
    metrics.update({
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'ic': float(ic),
        'rank_ic': float(rank_ic),
        'direction_accuracy': float(direction_accuracy),
    })
    return metrics


# ============================================
# 学习率调度器
# ============================================

def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('cosine', 'step', 'none')
        num_epochs: 总训练轮数
        warmup_epochs: 预热轮数
        min_lr: 最小学习率
        
    Returns:
        学习率调度器
    """
    if scheduler_type == "none":
        return None
    
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
            eta_min=min_lr,
        )
    
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1,
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# ============================================
# 日志记录
# ============================================

class Logger:
    """简单的日志记录器"""
    
    def __init__(self, log_dir: str = "./logs"):
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"train_{timestamp}.log")
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': [],
        }
    
    def log(self, message: str, print_msg: bool = True):
        """记录并打印消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        
        if print_msg:
            print(formatted)
        
        with open(self.log_file, 'a') as f:
            f.write(formatted + '\n')
    
    def log_epoch(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: float, 
        metrics: Dict[str, float],
        lr: float,
    ):
        """记录epoch信息"""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['metrics'].append(metrics)
        
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | LR: {lr:.6f} | {metrics_str}"
        )


# ============================================
# 早停
# ============================================

class EarlyStopping:
    """早停机制"""
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.0,
        mode: str = 'min',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前分数
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


# ============================================
# 模型保存和加载
# ============================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device,
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, float]:
    """加载模型检查点"""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, checkpoint['epoch'], checkpoint['loss']


if __name__ == "__main__":
    # 测试工具函数
    print("=" * 60)
    print("工具函数测试")
    print("=" * 60)
    
    # 测试设备选择
    device = get_device()
    print(f"使用设备: {device}")
    
    # 测试损失函数
    predictions = torch.randn(100)
    targets = torch.randn(100)
    
    mse_loss = get_loss_function('mse')
    ql_loss = get_loss_function('quasi_likelihood')
    ic_loss = get_loss_function('ic')
    
    print(f"\nMSE Loss: {mse_loss(predictions, targets):.4f}")
    print(f"Quasi-Likelihood Loss: {ql_loss(predictions, targets):.4f}")
    print(f"IC Loss: {ic_loss(predictions, targets):.4f}")
    
    # 测试评估指标
    print("\n评估指标:")
    metrics = compute_metrics(predictions.numpy(), targets.numpy())
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # 测试早停
    print("\n早停测试:")
    early_stopping = EarlyStopping(patience=3)
    losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88]
    for i, loss in enumerate(losses):
        should_stop = early_stopping(loss)
        print(f"  Epoch {i+1}, Loss: {loss:.2f}, Stop: {should_stop}")
