#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .base import LoRALayer


class mLoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lambda_num: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        tunable_scaler: bool = False,
        logging: bool = False,
        init_method: str = "default",  # Method to initialize the weights
        **kwargs
    ):
        """_summary_

        Parameters
        ----------
        in_features : int. The number of input features
        out_features : int. The number of output features
        B_num : int. The number of B matrices
        lambda_num : int. The number of lambda matrices (e.g., task number)
        diagonal_format : bool. Whether the lambda matrices are diagonal
        B_scale : float, optional. The scale of the B matrices. (e.g., tenpearature)
        r : int, optional. The rank of the LoRA decomposition
        lora_alpha : int, optional. The scaling factor for the LoRA decomposition
        lora_dropout : float, optional. The dropout rate for the LoRA decomposition
        fan_in_fan_out : bool, optional. Whether the layer stores the weight in fan_in, fan_out format
        tunable_scaler : bool, optional. Whether to use a tunable scaler
        """
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
            tunable_scaler=tunable_scaler,
        )
        self.init_method = init_method
        self.fan_in_fan_out = fan_in_fan_out
        self.lambda_num = lambda_num

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((lambda_num, out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        self.logging = logging  # Flag to control logging of statistics
        if self.logging:
            self._iteration = 0
            self.register_full_backward_hook(self._logging_hook)

    def _logging_hook(self, *_):
        if (self._iteration + 1) % 100 == 0:
            print(f"Iteration {self._iteration + 1}: lora_B_w: {self.lora_B_w.data}")
        self._iteration += 1

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            if self.init_method == "default":
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
            elif self.init_method == "bert":
                nn.init.trunc_normal_(self.lora_A, std=0.02)
                nn.init.trunc_normal_(self.lora_B, std=0.02)
            else:
                raise ValueError(f"Unknown initialization method: {self.init_method}")

    def forward(self, x: torch.Tensor, lambda_index: int, statistics=None):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        lora_A = self.lora_A
        lora_B = self.lora_B[lambda_index]

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            dropout_x = self.lora_dropout(x)
            after_A = dropout_x @ lora_A.transpose(-2, -1)
            after_B = after_A @ lora_B.transpose(-2, -1)
            result += (
                after_B * self.scaling * self.compute_tunable_scale(requires_grad=False)
            )
        if statistics is not None:
            statistics["after_A"] = after_A
            statistics["after_B"] = after_B
        return result
