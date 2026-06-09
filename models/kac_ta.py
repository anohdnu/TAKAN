import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        if self.init_scale == 0:
            nn.init.zeros_(self.weight)
        else:
            nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float = 2.,
        num_grids: int = 8,
        denominator: float = None,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class KACLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -3.,
        grid_max: float = 3.,
        num_grids: int = 16,
        num_tasks: int = 5,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(input_dim)

        step = (grid_max - grid_min) / (num_grids - 1)
        self._rbf_step = step
        self.num_tasks = num_tasks

        # width factors divided across tasks
        if num_tasks <= 1:
            self._width_factors = [1.0]
        else:
            self._width_factors = torch.linspace(0.5, 1.0, steps=num_tasks).tolist() 

        self._task_id = None

        # initialize with first task width
        self.rbf = RadialBasisFunction(
            grid_min,
            grid_max,
            num_grids,
            denominator=self._rbf_step * self._width_factors[0]
        )
        
        self.basis_linear = torch.nn.Parameter(torch.zeros([input_dim, num_grids]))
        nn.init.trunc_normal_(self.basis_linear, mean=0, std=spline_weight_init_scale)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, 0)

        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def set_task_id(self, task_id: int | None):
        self._task_id = task_id

    def _apply_task_width(self):
        if self._task_id is None:
            return
        idx = int(self._task_id)
        if idx < 0:
            return
        if idx >= len(self._width_factors):
            idx = len(self._width_factors) - 1
        self.rbf.denominator = self._rbf_step * self._width_factors[idx]

    def forward(self, x, time_benchmark=False):
        self._apply_task_width()

        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)

        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        return ret