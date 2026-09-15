# GCRSS

算法代码包。包含 Python 源文件和本 README。

## 文件

- `run_gcrss.py`：公开入口 `select_gcrss(X_train, m)`，内部仅用训练集完成零方差剔除与标准化。
- `methods/gcrss.py`：冻结 reusable-ranking 前端、建图和特征排序。
- `methods/reusable_ranking.py`：reusable ranking 参数与前端实现。
- `methods/ranking_core.py`：建图、Laplace score、相关性和排序基础函数。
- `methods/gcrss_stage2.py`：budget-conditioned subset realization 的约束交换。
- `methods/__init__.py`：Python 包声明。

## 最终配置

- graph scales: `k = {8, 12, 16}`；reference scale: `k = 12`；
- initialization: frozen reusable ranking top-`m`；
- candidate pool: top-500（特征不足时使用实际维度）；
- structural-consistency constraint: `g(S) <= 1.01 g(S0)`；
- refinement: deterministic best-improvement 1-for-1；
- maximum accepted swaps: `2`；
- selector 不读取 labels、test data 或 ACC/NMI。

## 用法

```python
import numpy as np
from run_gcrss import select_gcrss

# 只传入训练矩阵；行是样本，列是特征。不要传入标签或测试集。
X_train = np.asarray(..., dtype=float)
selected_features, diagnostics = select_gcrss(X_train, m=20)
print(selected_features)                 # 输入矩阵中的 0-based 列索引
print(diagnostics["accepted_swaps"])    # 0、1 或 2
```

安装依赖：`numpy`、`scipy`、`scikit-learn`。推荐 Python 3.9+。
