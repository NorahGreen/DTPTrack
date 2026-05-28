# https://huggingface.co/moonshotai/MoonViT-SO-400M/blob/main/modeling_moonvit.py
import torch
import torch.nn as nn


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding with multi-resolution support.
    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.
    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py
    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

        self.freqs_cis = None

    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.
        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(self, grid_hws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_hws (torch.Tensor): grid height and width
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_freqs_cis(grid_hws.device)

        shapes = grid_hws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [self.freqs_cis[:h, :w].reshape(-1, self.dim // 2) for h, w in shapes],
            dim=0,
        )
        return freqs_cis

def _apply_rope_input_validation(x, freqs_cis, num_heads_axis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    x_shape = list(x.shape)
    x_shape.pop(num_heads_axis)
    freqs_cis_shape = list(freqs_cis.shape)
    assert x_shape[:-1] == freqs_cis_shape[:-1], (x.shape, freqs_cis.shape)
    assert x_shape[-1] == 2 * freqs_cis_shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def apply_rope(
    x: torch.Tensor, freqs_cis: torch.Tensor, num_heads_axis: int = -2
) -> torch.Tensor:
    """
    Args: (The leading dimensions of all inputs should be the same)
        x: tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        x_out: tensor of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(x, freqs_cis, num_heads_axis)

    freqs_cis = freqs_cis.unsqueeze(num_heads_axis)  # ..., 1, head_dim/2
    # ..., num_heads, head_dim/2
    x_ = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return x_out.type_as(x)


def compute_freqs_cis(dim, max_height, max_width, theta_base=10000):
    """Calculate the cis(freqs) for each position in the 2D grid.
    Return: complex tensor of shape (max_height, max_width, dim//2) and value:
        height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
        weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
        note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
    """
    assert dim % 4 == 0, "dim must be divisible by 4"
    N = max_height * max_width
    flat_pos = torch.arange(0, N).float()
    x_pos = flat_pos % max_width
    y_pos = flat_pos // max_width
    dim_range = (
        torch.arange(0, dim, 4)[: (dim // 4)].float()
    )  # C/4
    freqs = 1.0 / (theta_base ** (dim_range / dim))
    x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
    y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
    # N, C/4, 2
    freqs_cis = torch.cat(
        [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
    )
    # max_height, max_width, C/2
    freqs_cis = freqs_cis.reshape(max_height, max_width, -1)
    return freqs_cis
