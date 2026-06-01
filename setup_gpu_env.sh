#!/usr/bin/env bash
set -euo pipefail
cd /home/scx/gat_quant

TORCH_VER=${TORCH_VER:-2.7.0}
CUDA_TAG=${CUDA_TAG:-cu128}

conda create -n gat_quant python=3.11 -y
conda activate gat_quant

pip install -U pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple

# PyTorch GPU 版
pip install torch==${TORCH_VER}+${CUDA_TAG} torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/${CUDA_TAG}

# PyTorch Geometric 主包
pip install torch_geometric -i https://pypi.tuna.tsinghua.edu.cn/simple

# PyG 底层扩展（必须与 torch 版本严格匹配）
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html

# 项目依赖
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    pandas numpy scipy scikit-learn tqdm pyyaml matplotlib seaborn pytest \
    "tushare>=1.4" python-dotenv

# 验证
python - <<'PY'
import sys, torch
print('python', sys.version)
print('torch', torch.__version__)
print('torch cuda', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device count', torch.cuda.device_count())
    print('device0', torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device='cuda')
    y = x @ x
    print('matmul ok', float(y[0, 0].detach().cpu()))

import torch_geometric
print('torch_geometric', torch_geometric.__version__)

from torch_geometric.data import HeteroData
d = HeteroData()
d['a'].x = torch.zeros(3, 4)
print('HeteroData ok')
PY
