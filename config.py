"""
配置管理模块
Configuration Management Module

包含所有超参数和系统配置
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


@dataclass
class DataConfig:
    """数据配置"""
    num_stocks: int = 0             # 股票节点数量（0 = 使用全部可用股票）
    num_sectors: int = 8            # 板块节点数量（行业映射存在时由实际行业数决定）


@dataclass
class ModelConfig:
    """模型配置 - 基于论文推荐参数"""
    # 隐藏层维度
    hidden_dim: int = 256           # 平衡模型容量与过拟合风险
    
    # 注意力参数
    num_heads: int = 4              # 多头注意力头数（4-8）
    
    # GNN层数
    num_gnn_layers: int = 2         # 2层足以覆盖二阶邻居，避免过平滑
    
    # 正则化
    dropout: float = 0.3            # Dropout率（0.3-0.5，金融数据噪声大）
    
    # 输出维度
    output_dim: int = 1             # 预测收益率（单值输出）


@dataclass
class TrainingConfig:
    """训练配置"""
    # 学习率
    learning_rate: float = 1e-4     # 配合AdamW使用
    weight_decay: float = 1e-5      # L2正则化
    
    # 训练参数
    batch_size: int = 16
    num_epochs: int = 100
    
    # 学习率调度
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # 余弦退火调度
    warmup_epochs: int = 5
    
    # 早停
    early_stopping_patience: int = 10
    
    # 损失函数
    loss_type: str = "mse"          # 可选: "mse", "quasi_likelihood"
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # 随机种子
    seed: int = 42


@dataclass
class Config:
    """总配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 项目路径
    project_name: str = "gat_quant"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    
    def __post_init__(self):
        """配置后处理"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# 默认配置实例
default_config = Config()


def get_config() -> Config:
    """获取默认配置"""
    return Config()
