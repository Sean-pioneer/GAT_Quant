"""
异构图卷积层
Heterogeneous Graph Convolution Layer

实现论文中的两级注意力机制:
1. 节点级注意力 (Intra-Metapath Aggregation): 每种边类型内的消息传递
2. 语义级注意力 (Inter-Metapath Aggregation): 融合不同边类型的信息

元路径 (Meta-path) 说明:
- 股票-板块-股票: 同行业关联
- 股票-股票 (相关性): 价格联动关联 (Pearson |r| ≥ corr_threshold)
- 股票-股票 (波动率溢出): 风险传染关联 (lag-1 cross-correlation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple, Union

class HeteroGATConv(nn.Module):
    """
    异构图注意力卷积层
    
    实现两级注意力机制:
    
    Level 1 - 节点级注意力 (Intra-Metapath):
    =========================================
    对于特定的元路径Φ，计算目标节点i与邻居节点j之间的注意力系数:
    
        e^Φ_ij = a^T · LeakyReLU(W^Φ · [h_i || h_j])
        α^Φ_ij = softmax_j(e^Φ_ij) · σ(P_j)  # 带人气门控
        
        h^Φ_i = Σ_j α^Φ_ij · W^Φ · h_j
    
    Level 2 - 语义级注意力 (Inter-Metapath):
    =========================================
    学习不同元路径的重要性权重:
    
        w^Φ = tanh(b · mean(h^Φ))
        β^Φ = softmax_Φ(w^Φ)
        
        h^{struct}_i = Σ_Φ β^Φ · h^Φ_i
    
    参数:
    ======
    in_channels: 输入特征维度 (字典形式)
    hidden_channels: 隐藏层维度
    out_channels: 输出特征维度
    heads: 注意力头数
    dropout: Dropout率
    """
    
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        hidden_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # 处理输入维度
        if isinstance(in_channels, int):
            stock_in = in_channels
            sector_in = in_channels
        else:
            stock_in = in_channels.get('stock', 30)
            sector_in = in_channels.get('sector', 30)
        
        feature_dim = hidden_channels * heads
        self._feat_dim = feature_dim

        # Level 1：每种边类型独立的 GATv2Conv
        self.conv_belongs_to = GATv2Conv(          # stock → sector
            in_channels=(stock_in, sector_in),
            out_channels=hidden_channels, heads=heads,
            dropout=dropout, add_self_loops=False, concat=True,
        )
        self.conv_contains = GATv2Conv(            # sector → stock
            in_channels=(sector_in, stock_in),
            out_channels=hidden_channels, heads=heads,
            dropout=dropout, add_self_loops=False, concat=True,
        )
        self.conv_correlates = GATv2Conv(          # stock → stock（相关性）
            in_channels=stock_in,
            out_channels=hidden_channels, heads=heads,
            dropout=dropout, add_self_loops=False, concat=True,
        )
        self.conv_spills = GATv2Conv(              # stock → stock（溢出）
            in_channels=stock_in,
            out_channels=hidden_channels, heads=heads,
            dropout=dropout, add_self_loops=False, concat=True,
        )

        # Level 2：语义级注意力 — 对 stock 节点的 3 条入路径学习重要性权重
        self.stock_metapath_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.Tanh(),
            nn.Linear(feature_dim // 4, 1),
        )

        self.sector_projection = nn.Linear(feature_dim, out_channels)
        self.stock_projection  = nn.Linear(feature_dim, out_channels)
        self.stock_norm  = nn.LayerNorm(out_channels)
        self.sector_norm = nn.LayerNorm(out_channels)
    
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        popularity_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            x_dict: 节点特征字典
                - 'stock': [num_stocks, in_channels]
                - 'sector': [num_sectors, in_channels]
            edge_index_dict: 边索引字典
            popularity_dict: 人气值字典 (可选)
                - 'stock': [num_stocks]

        Returns:
            out_dict: 更新后的节点特征字典
                - 'stock': [num_stocks, out_channels]
                - 'sector': [num_sectors, out_channels]
        """
        stock_x  = x_dict['stock']
        sector_x = x_dict['sector']
        num_stocks  = stock_x.size(0)
        num_sectors = sector_x.size(0)
        fd = self._feat_dim

        def _run(conv, src, dst, ei, n_dst):
            """空边时返回零向量，避免 GATv2Conv 报错。"""
            if ei.size(1) == 0:
                return src.new_zeros(n_dst, fd)
            return conv((src, dst), ei) if dst is not None else conv(src, ei)

        ei = edge_index_dict  # 简写

        # Level 1：每条路径独立聚合
        h_sector = _run(self.conv_belongs_to, stock_x,  sector_x,
                        ei[('stock', 'belongs_to', 'sector')], num_sectors)

        h1 = _run(self.conv_contains,   sector_x, stock_x,
                  ei[('sector', 'contains', 'stock')], num_stocks)         # sector→stock
        h2 = _run(self.conv_correlates, stock_x,  None,
                  ei[('stock', 'correlates_with', 'stock')], num_stocks)   # 相关性
        h3 = _run(self.conv_spills,     stock_x,  None,
                  ei[('stock', 'spills_volatility_to', 'stock')], num_stocks)  # 溢出

        # Level 2：语义注意力 — 对 3 条路径学习重要性权重
        paths = torch.stack([h1, h2, h3], dim=1)              # [N, 3, fd]
        N = paths.size(0)
        scores = self.stock_metapath_attention(
            paths.view(N * 3, fd)                              # [N*3, fd]
        ).view(N, 3, 1)                                        # [N, 3, 1]
        weights = torch.softmax(scores, dim=1)                 # [N, 3, 1]
        h_stock = (weights * paths).sum(dim=1)                 # [N, fd]

        # 投影 + 归一化 + 激活
        return {
            'stock':  F.elu(self.stock_norm(self.stock_projection(h_stock))),
            'sector': F.elu(self.sector_norm(self.sector_projection(h_sector))),
        }


class StackedHeteroGATConv(nn.Module):
    """
    堆叠的异构图注意力层
    
    多层HeteroGATConv的堆叠，论文建议使用2层以覆盖二阶邻居
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        
        # 第一层
        self.convs.append(
            HeteroGATConv(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=heads,
                dropout=dropout,
            )
        )
        
        # 后续层
        for _ in range(num_layers - 1):
            self.convs.append(
                HeteroGATConv(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    out_channels=out_channels if _ == num_layers - 2 else hidden_channels,
                    heads=heads,
                    dropout=dropout,
                )
            )
    
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        popularity_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        前向传播
        
        Args:
            x_dict: 节点特征字典
            edge_index_dict: 边索引字典
            popularity_dict: 人气值字典
            
        Returns:
            out_dict: 更新后的节点特征字典
        """
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict, popularity_dict)
            
            if i < self.num_layers - 1:
                # 应用dropout（除了最后一层）
                x_dict = {
                    key: F.dropout(val, p=self.dropout, training=self.training)
                    for key, val in x_dict.items()
                }
        
        return x_dict


if __name__ == "__main__":
    # 测试异构图卷积
    print("=" * 60)
    print("HeteroGATConv 测试")
    print("=" * 60)
    
    # 创建测试数据
    num_stocks = 100
    num_sectors = 10
    in_channels = 30
    hidden_channels = 64
    out_channels = 128
    
    x_dict = {
        'stock': torch.randn(num_stocks, in_channels),
        'sector': torch.randn(num_sectors, in_channels),
    }
    
    # 创建边索引
    edge_index_dict = {
        ('stock', 'belongs_to', 'sector'): torch.stack([
            torch.arange(num_stocks),
            torch.randint(0, num_sectors, (num_stocks,))
        ]),
        ('sector', 'contains', 'stock'): torch.stack([
            torch.randint(0, num_sectors, (num_stocks,)),
            torch.arange(num_stocks)
        ]),
        ('stock', 'spills_volatility_to', 'stock'): torch.randint(
            0, num_stocks, (2, 500)
        ),
        ('stock', 'correlates_with', 'stock'): torch.randint(
            0, num_stocks, (2, 300)
        ),
    }
    
    # 测试单层
    layer = HeteroGATConv(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=4,
    )
    
    out_dict = layer(x_dict, edge_index_dict)
    
    print("输入形状:")
    for key, val in x_dict.items():
        print(f"  {key}: {val.shape}")
    
    print("\n输出形状:")
    for key, val in out_dict.items():
        print(f"  {key}: {val.shape}")
    
    # 测试堆叠层
    print("\n" + "=" * 60)
    print("StackedHeteroGATConv 测试")
    print("=" * 60)
    
    model = StackedHeteroGATConv(
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=128,
        num_layers=2,
        heads=4,
    )
    
    out_dict = model(x_dict, edge_index_dict)
    
    print("堆叠层输出形状:")
    for key, val in out_dict.items():
        print(f"  {key}: {val.shape}")
