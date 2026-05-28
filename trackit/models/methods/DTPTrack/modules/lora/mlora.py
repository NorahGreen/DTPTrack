#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn
from . import LoRALayer


class mLoRALinear(nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        task_r: int,
        num_tasks: int,
        init_method: str = "bert",
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.lora = LoRALayer(in_features, out_features, r, r, dropout=0, rs_lora=False, init_method=init_method)
        self.task_loras = nn.ModuleList(
            [
                LoRALayer(in_features, out_features, task_r, task_r, dropout=0, rs_lora=False, init_method=init_method)
                for _ in range(num_tasks)
            ]
        )

    def forward(self, x: torch.Tensor, task_idx: int):
        result = F.linear(x, self.weight, bias=self.bias)
        return result + 0.5 * self.lora(x) + 0.5 * self.task_loras[task_idx](x)
