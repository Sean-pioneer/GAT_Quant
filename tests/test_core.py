"""Core unit tests for MarketSentimentGAT.

Run from project root:
    python -m pytest tests/test_core.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData


# ─────────────────────────────────────────────
# 1. Feature dimension constants
# ─────────────────────────────────────────────

class TestFeatureDimensions:
    def test_node_feature_columns_count(self):
        from data.daily_report_dataset import NODE_FEATURE_COLUMNS, NODE_FEATURE_DIM
        assert len(NODE_FEATURE_COLUMNS) == NODE_FEATURE_DIM, (
            f"NODE_FEATURE_COLUMNS has {len(NODE_FEATURE_COLUMNS)} entries "
            f"but NODE_FEATURE_DIM={NODE_FEATURE_DIM}"
        )

    def test_node_feature_dim_value(self):
        from data.daily_report_dataset import NODE_FEATURE_DIM
        assert NODE_FEATURE_DIM == 18

    def test_no_duplicate_columns(self):
        from data.daily_report_dataset import NODE_FEATURE_COLUMNS
        assert len(NODE_FEATURE_COLUMNS) == len(set(NODE_FEATURE_COLUMNS)), \
            "Duplicate entries in NODE_FEATURE_COLUMNS"


# ─────────────────────────────────────────────
# 2. Helper functions
# ─────────────────────────────────────────────

class TestNormalizeStockCode:
    def setup_method(self):
        from data.daily_report_dataset import _normalize_stock_code
        self.fn = _normalize_stock_code

    def test_plain_6digit(self):
        assert self.fn("600519") == "600519"

    def test_float_suffix(self):
        assert self.fn("600519.0") == "600519"

    def test_short_code_zero_padded(self):
        assert self.fn("1") == "000001"

    def test_tushare_code_sh(self):
        assert self.fn("600519.SH") == "600519"

    def test_tushare_code_sz(self):
        assert self.fn("000001.SZ") == "000001"

    def test_nan_returns_empty(self):
        assert self.fn(float("nan")) == ""

    def test_integer_input(self):
        assert self.fn(600519) == "600519"


class TestBoardFromCode:
    def setup_method(self):
        from data.daily_report_dataset import _board_from_code
        self.fn = _board_from_code

    def test_star_market(self):
        assert self.fn("688001") == "STAR"

    def test_chinext(self):
        assert self.fn("300001") == "ChiNext"
        assert self.fn("301001") == "ChiNext"

    def test_sz_main(self):
        assert self.fn("000001") == "SZ_Main"
        assert self.fn("002001") == "SZ_Main"

    def test_sh_main(self):
        assert self.fn("600519") == "SH_Main"
        assert self.fn("601318") == "SH_Main"

    def test_unknown_falls_to_other(self):
        assert self.fn("430001") == "Other"


class TestRobustCenterScale:
    def setup_method(self):
        from data.daily_report_dataset import _robust_center_scale
        self.fn = _robust_center_scale

    def test_normal_series(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        center, scale = self.fn(s)
        assert center == pytest.approx(3.0, abs=1e-6)
        assert scale > 0

    def test_constant_series_returns_scale_one(self):
        s = pd.Series([7.0] * 10)
        center, scale = self.fn(s)
        assert center == pytest.approx(7.0)
        assert scale == pytest.approx(1.0)

    def test_all_nan_returns_defaults(self):
        s = pd.Series([float("nan")] * 5)
        center, scale = self.fn(s)
        assert center == 0.0
        assert scale == 1.0

    def test_ignores_inf(self):
        s = pd.Series([1.0, 2.0, float("inf"), 3.0, 4.0])
        center, scale = self.fn(s)
        assert np.isfinite(center)
        assert np.isfinite(scale)

    def test_scale_positive(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 100))
        _, scale = self.fn(s)
        assert scale > 0


# ─────────────────────────────────────────────
# 3. Model layers
# ─────────────────────────────────────────────

def _make_hetero_data(num_stocks: int = 10, num_sectors: int = 4,
                      node_dim: int = 18) -> HeteroData:
    """Build a minimal synthetic HeteroData for testing."""
    data = HeteroData()
    data["stock"].x = torch.randn(num_stocks, node_dim)
    data["sector"].x = torch.randn(num_sectors, node_dim)

    # belongs_to edges: each stock → one sector
    stock_ids = torch.arange(num_stocks)
    sector_ids = stock_ids % num_sectors
    data["stock", "belongs_to", "sector"].edge_index = torch.stack([stock_ids, sector_ids])
    data["sector", "contains", "stock"].edge_index = torch.stack([sector_ids, stock_ids])

    # correlates_with / spills_volatility_to: a few stock→stock edges
    src = torch.tensor([0, 1, 2, 3])
    dst = torch.tensor([1, 2, 3, 0])
    data["stock", "correlates_with", "stock"].edge_index = torch.stack([src, dst])
    data["stock", "spills_volatility_to", "stock"].edge_index = torch.stack([src, dst])

    return data


class TestHeteroGATConv:
    def test_forward_output_shape(self):
        from models.layers.hetero_conv import HeteroGATConv
        num_stocks, num_sectors, node_dim, hidden_dim = 10, 4, 21, 64
        data = _make_hetero_data(num_stocks, num_sectors, node_dim)

        layer = HeteroGATConv(
            in_channels=node_dim,
            hidden_channels=hidden_dim // 4,
            out_channels=hidden_dim,
            heads=4,
            dropout=0.0,
        )
        layer.eval()
        x_dict = {"stock": data["stock"].x, "sector": data["sector"].x}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

        out = layer(x_dict, edge_index_dict)
        assert "stock" in out
        assert out["stock"].shape == (num_stocks, hidden_dim)

    def test_no_nan_in_output(self):
        from models.layers.hetero_conv import HeteroGATConv
        data = _make_hetero_data()
        layer = HeteroGATConv(in_channels=18, hidden_channels=16, out_channels=64,
                              heads=4, dropout=0.0)
        layer.eval()
        x_dict = {"stock": data["stock"].x, "sector": data["sector"].x}
        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}
        out = layer(x_dict, edge_index_dict)
        assert not torch.isnan(out["stock"]).any(), "NaN in HeteroGATConv output"



# ─────────────────────────────────────────────
# 4. Full model forward pass
# ─────────────────────────────────────────────

class TestMarketSentimentGAT:
    def _build_model(self, node_dim=18):
        from models.market_sentiment_gat import MarketSentimentGAT
        return MarketSentimentGAT(
            node_feature_dim=node_dim,
            hidden_dim=64,
            num_heads=4,
            num_gnn_layers=2,
            dropout=0.0,
            output_dim=1,
        )

    def test_output_shape_single_graph(self):
        model = self._build_model()
        model.eval()
        num_stocks = 10
        data = _make_hetero_data(num_stocks=num_stocks)
        with torch.no_grad():
            out = model(data)
        assert out.shape == (1, num_stocks), f"Expected (1, {num_stocks}), got {out.shape}"

    def test_no_nan_in_predictions(self):
        model = self._build_model()
        model.eval()
        data = _make_hetero_data(num_stocks=10)
        with torch.no_grad():
            out = model(data)
        assert not torch.isnan(out).any(), "NaN in model predictions"

    def test_gradient_flows(self):
        model = self._build_model()
        model.train()
        num_stocks = 8
        data = _make_hetero_data(num_stocks=num_stocks)
        targets = torch.randn(1, num_stocks)
        out = model(data)
        loss = torch.nn.functional.mse_loss(out, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_different_stock_counts(self):
        model = self._build_model()
        model.eval()
        for n in [5, 20, 50]:
            data = _make_hetero_data(num_stocks=n, num_sectors=3)
            with torch.no_grad():
                out = model(data)
            assert out.shape == (1, n)
