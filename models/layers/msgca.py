"""
多模态门控交叉注意力融合层 (MSGCA)
Multimodal Stable Fusion with Gated Cross-Attention

实现论文中的MSGCA机制，用于融合不同模态的特征:
- 主模态 (Primary): 量价技术指标 (稳定性高)
- 辅助模态 (Auxiliary): GAT图结构特征 (可能有噪声)

核心思想:
=========
利用"主模态"来引导"辅助模态"的融合，通过门控机制过滤噪声

数学公式:
=========
1. 交叉注意力 (Cross-Attention):
   以指标特征为Query，图特征为Key和Value
   
   CA(Q, K, V) = softmax(QK^T / √d_k) V
   
   其中:
   - Q = W_q · F^{ind}  (来自技术指标)
   - K = W_k · F^{graph} (来自图特征)
   - V = W_v · F^{graph} (来自图特征)

2. 门控机制 (Gating):
   门控信号完全由稳定的主模态生成
   
   G = σ(W_g · F^{ind})
   
   F^{cross} = CA(Q, K, V)
   F^{gated} = G ⊙ F^{cross}

3. 残差连接:
   保留原始的主模态特征
   
   F^{fused} = F^{gated} + F^{ind}

优势:
=====
- 模型仅在技术指标认为"需要参考市场结构"时，才会引入GAT的特征
- 极大增强了模型的抗噪能力
- 解决了多模态数据中的语义冲突问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math


class MSGCAFusion(nn.Module):
    """
    多模态门控交叉注意力融合层
    
    维度流向:
    =========
    输入:
    - primary_features (技术指标): [batch_size, seq_len, feature_dim] 或
                                   [batch_size, num_nodes, seq_len, feature_dim]
    - auxiliary_features (图特征): [batch_size, seq_len, feature_dim] 或
                                   [batch_size, num_nodes, feature_dim]
    
    输出:
    - fused_features: [batch_size, feature_dim] 或
                      [batch_size, num_nodes, feature_dim]
    
    参数:
    ======
    primary_dim: 主模态特征维度
    auxiliary_dim: 辅助模态特征维度
    hidden_dim: 隐藏层维度
    num_heads: 多头注意力头数
    dropout: Dropout率
    use_residual: 是否使用残差连接
    """
    
    def __init__(
        self,
        primary_dim: int,
        auxiliary_dim: int,
        hidden_dim: int,
        num_heads: int = 2,
        dropout: float = 0.3,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.primary_dim = primary_dim
        self.auxiliary_dim = auxiliary_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # ============================================
        # 投影层: 将不同模态映射到相同的隐藏空间
        # ============================================
        
        # 主模态投影 (用于生成Query和门控信号)
        self.primary_projection = nn.Linear(primary_dim, hidden_dim)
        
        # 辅助模态投影 (用于生成Key和Value)
        self.auxiliary_projection = nn.Linear(auxiliary_dim, hidden_dim)
        
        # ============================================
        # 交叉注意力的Q, K, V投影
        # ============================================
        self.q_projection = nn.Linear(hidden_dim, hidden_dim)
        self.k_projection = nn.Linear(hidden_dim, hidden_dim)
        self.v_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.out_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # ============================================
        # 门控机制
        # ============================================
        self.gate_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        # ============================================
        # 层归一化和Dropout
        # ============================================
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # ============================================
        # 前馈网络 (可选)
        # ============================================
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # 输出维度
        self.output_dim = hidden_dim
    
    def _reshape_for_attention(
        self, 
        x: Tensor, 
        batch_size: int
    ) -> Tensor:
        """
        重塑张量用于多头注意力
        
        [batch_size, seq_len, hidden_dim]
        -> [batch_size, num_heads, seq_len, head_dim]
        """
        seq_len = x.size(1)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def _compute_cross_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算交叉注意力
        
        Args:
            query: [batch_size, num_heads, query_len, head_dim]
            key: [batch_size, num_heads, key_len, head_dim]
            value: [batch_size, num_heads, value_len, head_dim]
            mask: 可选的注意力掩码
            
        Returns:
            output: [batch_size, query_len, hidden_dim]
        """
        batch_size = query.size(0)
        query_len = query.size(2)
        
        # 计算注意力分数
        # [batch_size, num_heads, query_len, key_len]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 加权求和
        # [batch_size, num_heads, query_len, head_dim]
        context = torch.matmul(attention_weights, value)
        
        # 重塑回原始形状
        # [batch_size, query_len, hidden_dim]
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, query_len, self.hidden_dim)
        
        # 输出投影
        output = self.out_projection(context)
        
        return output
    
    def forward(
        self,
        primary_features: Tensor,
        auxiliary_features: Tensor,
        return_attention: bool = False,
    ) -> Tensor:
        """
        前向传播
        
        Args:
            primary_features: 主模态特征 (技术指标序列)
                - [batch_size, seq_len, primary_dim] 或
                - [batch_size, num_nodes, seq_len, primary_dim]
            auxiliary_features: 辅助模态特征 (图结构特征)
                - [batch_size, seq_len, auxiliary_dim] 或
                - [batch_size, num_nodes, auxiliary_dim]
            return_attention: 是否返回注意力权重
            
        Returns:
            fused_features: 融合后的特征
        """
        # 处理不同输入维度
        original_shape = primary_features.shape
        has_node_dim = len(original_shape) == 4
        
        if has_node_dim:
            # [batch_size, num_nodes, seq_len, primary_dim]
            batch_size, num_nodes, seq_len, _ = primary_features.shape
            
            # 展平节点维度
            primary_features = primary_features.view(
                batch_size * num_nodes, seq_len, -1
            )
            
            if len(auxiliary_features.shape) == 3:
                # [batch_size, num_nodes, auxiliary_dim]
                # -> [batch_size * num_nodes, 1, auxiliary_dim]
                auxiliary_features = auxiliary_features.view(
                    batch_size * num_nodes, 1, -1
                )
            else:
                # [batch_size, num_nodes, seq_len, auxiliary_dim]
                auxiliary_features = auxiliary_features.view(
                    batch_size * num_nodes, seq_len, -1
                )
            
            effective_batch_size = batch_size * num_nodes
        else:
            batch_size, seq_len, _ = primary_features.shape
            effective_batch_size = batch_size
            
            if len(auxiliary_features.shape) == 2:
                # [batch_size, auxiliary_dim] -> [batch_size, 1, auxiliary_dim]
                auxiliary_features = auxiliary_features.unsqueeze(1)
        
        # ============================================
        # Step 1: 投影到隐藏空间
        # ============================================
        primary_hidden = self.primary_projection(primary_features)
        # [batch_size, seq_len, hidden_dim]
        
        auxiliary_hidden = self.auxiliary_projection(auxiliary_features)
        # [batch_size, aux_seq_len, hidden_dim]
        
        # ============================================
        # Step 2: 计算Q, K, V
        # ============================================
        # Query来自主模态 (取最后一个时间步或平均)
        # 使用最后一个时间步的特征作为query
        query_features = primary_hidden[:, -1:, :]  # [batch_size, 1, hidden_dim]
        
        Q = self.q_projection(query_features)
        K = self.k_projection(auxiliary_hidden)
        V = self.v_projection(auxiliary_hidden)
        
        # 重塑为多头格式
        Q = self._reshape_for_attention(Q, effective_batch_size)
        K = self._reshape_for_attention(K, effective_batch_size)
        V = self._reshape_for_attention(V, effective_batch_size)
        
        # ============================================
        # Step 3: 交叉注意力
        # ============================================
        cross_attention_output = self._compute_cross_attention(Q, K, V)
        # [batch_size, 1, hidden_dim]
        
        cross_attention_output = cross_attention_output.squeeze(1)
        # [batch_size, hidden_dim]
        
        # ============================================
        # Step 4: 门控机制
        # ============================================
        # 门控信号由主模态生成
        gate_input = primary_hidden[:, -1, :]  # [batch_size, hidden_dim]
        gate = self.gate_linear(gate_input)
        # [batch_size, hidden_dim]
        
        # 应用门控
        gated_output = gate * cross_attention_output
        # [batch_size, hidden_dim]
        
        # ============================================
        # Step 5: 残差连接
        # ============================================
        if self.use_residual:
            # 将主模态特征投影到相同维度后相加
            residual = primary_hidden[:, -1, :]  # [batch_size, hidden_dim]
            fused = self.layer_norm1(gated_output + residual)
        else:
            fused = self.layer_norm1(gated_output)
        
        # ============================================
        # Step 6: 前馈网络 + 残差
        # ============================================
        ffn_output = self.ffn(fused)
        fused = self.layer_norm2(ffn_output + fused)
        # [batch_size, hidden_dim]
        
        # ============================================
        # 恢复节点维度（如果需要）
        # ============================================
        if has_node_dim:
            fused = fused.view(batch_size, num_nodes, self.hidden_dim)
            # [batch_size, num_nodes, hidden_dim]
        
        return fused


class SimpleMSGCA(nn.Module):
    """
    简化版MSGCA融合
    
    使用PyTorch内置的MultiheadAttention，更简洁
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.gate_linear = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output_dim = embed_dim
    
    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
    ) -> Tensor:
        """
        前向传播
        
        Args:
            query: 主模态特征 [batch_size, query_len, embed_dim]
            key_value: 辅助模态特征 [batch_size, kv_len, embed_dim]
            
        Returns:
            output: 融合特征 [batch_size, query_len, embed_dim]
        """
        # 交叉注意力
        attn_output, _ = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value,
        )
        
        # 门控机制
        gate = torch.sigmoid(self.gate_linear(query))
        gated_output = gate * attn_output
        
        # 残差连接
        output = self.layer_norm(gated_output + query)
        
        return output


if __name__ == "__main__":
    # 测试MSGCA融合层
    print("=" * 60)
    print("MSGCAFusion 测试")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 20
    num_nodes = 100
    primary_dim = 20  # 技术指标维度 (OHLCV + 技术指标)
    auxiliary_dim = 128  # 图特征维度
    hidden_dim = 128
    
    # 测试普通输入
    primary = torch.randn(batch_size, seq_len, primary_dim)
    auxiliary = torch.randn(batch_size, seq_len, auxiliary_dim)
    
    msgca = MSGCAFusion(
        primary_dim=primary_dim,
        auxiliary_dim=auxiliary_dim,
        hidden_dim=hidden_dim,
        num_heads=4,
    )
    
    output = msgca(primary, auxiliary)
    print(f"普通输入:")
    print(f"  主模态形状: {primary.shape}")
    print(f"  辅助模态形状: {auxiliary.shape}")
    print(f"  输出形状: {output.shape}")
    
    # 测试带节点维度的输入
    print("\n" + "-" * 40)
    primary_nodes = torch.randn(batch_size, num_nodes, seq_len, primary_dim)
    auxiliary_nodes = torch.randn(batch_size, num_nodes, auxiliary_dim)
    
    output_nodes = msgca(primary_nodes, auxiliary_nodes)
    print(f"带节点维度的输入:")
    print(f"  主模态形状: {primary_nodes.shape}")
    print(f"  辅助模态形状: {auxiliary_nodes.shape}")
    print(f"  输出形状: {output_nodes.shape}")
    
    # 测试简化版
    print("\n" + "=" * 60)
    print("SimpleMSGCA 测试")
    print("=" * 60)
    
    simple_msgca = SimpleMSGCA(embed_dim=128, num_heads=4)
    
    query = torch.randn(batch_size, 1, 128)
    kv = torch.randn(batch_size, seq_len, 128)
    
    output_simple = simple_msgca(query, kv)
    print(f"Query形状: {query.shape}")
    print(f"Key/Value形状: {kv.shape}")
    print(f"输出形状: {output_simple.shape}")
