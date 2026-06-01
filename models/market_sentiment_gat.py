"""
市场情绪图注意力网络主模型
Market Sentiment Graph Attention Network

架构流程:
=========
输入: 单帧 HeteroData（stock + sector 节点，4 类边）

处理流程:
1. HeteroGATConv×2 提取异构图结构特征
2. MLP 预测收益率

输出: 收益率预测 [batch_size, num_stocks]
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData, Batch
from typing import Dict, Tuple, Union

from .layers import StackedHeteroGATConv


class MarketSentimentGAT(nn.Module):
    """
    市场情绪图注意力网络

    参数:
    ======
    node_feature_dim: 节点特征维度 (默认18)
    hidden_dim: 隐藏层维度 (默认256)
    num_heads: 注意力头数 (默认4)
    num_gnn_layers: GNN层数 (默认2)
    dropout: Dropout率 (默认0.3)
    output_dim: 输出维度 (默认1，预测收益率)

    维度变换:
    =========
    节点特征: [N, node_feature_dim]
    → HeteroGATConv×2: [N, hidden_dim]
    → MLP: [batch_size, num_stocks]
    """
    
    def __init__(
        self,
        node_feature_dim: int = 18,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_gnn_layers: int = 2,
        dropout: float = 0.05,
        output_dim: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.hetero_gat = StackedHeteroGATConv(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim // num_heads,
            out_channels=hidden_dim,
            num_layers=num_gnn_layers,
            heads=num_heads,
            dropout=dropout,
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def _extract_graph_features(
        self,
        data: Union[HeteroData, Batch],
    ) -> Tuple[Dict, Dict]:
        """从 HeteroData / Batch 中提取节点特征字典和边索引字典。"""
        x_dict = {
            'stock': data['stock'].x,
            'sector': data['sector'].x,
        }
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        return x_dict, edge_index_dict

    def forward(
        self,
        data: Union[HeteroData, Batch],
    ) -> Tensor:
        x_dict, edge_index_dict = self._extract_graph_features(data)
        out_dict = self.hetero_gat(x_dict, edge_index_dict)
        stock_emb = out_dict['stock']   # [total_stocks, hidden_dim]

        if isinstance(data, Batch):
            batch_size = data['stock'].batch.max().item() + 1
            num_stocks = stock_emb.size(0) // batch_size
        else:
            batch_size, num_stocks = 1, stock_emb.size(0)

        # BatchNorm1d 要求 2D 输入，先展平再还原
        predictions = self.predictor(stock_emb)                          # [total_stocks, output_dim]
        predictions = predictions.view(batch_size, num_stocks, -1)       # [B, N, output_dim]

        if self.output_dim == 1:
            predictions = predictions.squeeze(-1)   # [B, N]

        return predictions


class SimpleMarketGAT(nn.Module):
    """
    简化版市场GAT模型
    
    用于快速测试和调试，不需要完整的时序图序列
    """
    
    def __init__(
        self,
        node_feature_dim: int = 30,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
        output_dim: int = 1,
    ):
        super().__init__()
        
        self.hetero_gat = StackedHeteroGATConv(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim // num_heads,
            out_channels=hidden_dim,
            num_layers=2,
            heads=num_heads,
            dropout=dropout,
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(
        self,
        data: Union[HeteroData, Batch],
    ) -> Tensor:
        """
        前向传播
        
        Args:
            data: HeteroData对象
            
        Returns:
            predictions: 预测值
        """
        x_dict = {
            'stock': data['stock'].x,
            'sector': data['sector'].x,
        }
        
        edge_index_dict = {}
        for edge_type in data.edge_types:
            edge_index_dict[edge_type] = data[edge_type].edge_index
        
        out_dict = self.hetero_gat(x_dict, edge_index_dict)
        stock_emb = out_dict['stock']
        
        predictions = self.predictor(stock_emb)
        
        return predictions.squeeze(-1)


