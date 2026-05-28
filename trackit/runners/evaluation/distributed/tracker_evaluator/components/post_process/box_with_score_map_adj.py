import torch
import numpy as np
from trackit.core.operator.bbox.format import bbox_xyxy_to_cxcywh_torch
from typing import Tuple, Optional
from . import TrackerOutputPostProcess


class PostProcessing_BoxWithScoreMap_2(TrackerOutputPostProcess):
    def __init__(self, device: torch.device,
                 response_map_size: Tuple[int, int], # (H, W)
                 search_region_size: Tuple[int, int], # (H, W)
                 window_penalty_ratio: float = 0.0,
                 use_adaptive_penalty: bool = False,
                 adaptive_penalty_sigma_scale: float = 0.5, # Sigma = box_size * sigma_scale / response_size
                 classification_score_with_penalty: bool = False, # Flag kept for consistency, implemented below
                 min_sigma: float = 1.0 # Minimum sigma on the grid to avoid degenerate Gaussians
                 ):
        """
        Initializes the post-processing module.

        Args:
            device: The PyTorch device.
            response_map_size: (H, W) of the score map grid.
            search_region_size: (H, W) of the search region image patch.
            window_penalty_ratio: Weight (0.0 to 1.0) for the penalty. 0 disables penalty.
            use_adaptive_penalty: If True, use a Gaussian window centered on 'prior_box'.
                                 If False, use a fixed centered Hanning window.
            adaptive_penalty_sigma_scale: Factor scaling prior box size to Gaussian sigma
                                          on the response map grid. Smaller values mean
                                          a sharper, more localized window.
            classification_score_with_penalty: If True and penalty is enabled,
                                               the reported confidence score is the
                                               penalized score. Otherwise, it's the
                                               original score at the best location found
                                               using the penalized map.
            min_sigma: Minimum standard deviation for the adaptive Gaussian grid axes.
        """
        if not (0.0 <= window_penalty_ratio <= 1.0):
            raise ValueError("window_penalty_ratio must be between 0.0 and 1.0")

        self._response_map_size = response_map_size # (H, W)
        self._search_region_size = search_region_size # (H, W)
        self._device = device
        self._enable_penalty = window_penalty_ratio > 0.0
        self._window_penalty_ratio = window_penalty_ratio
        self._use_adaptive_penalty = use_adaptive_penalty and self._enable_penalty
        self._adaptive_penalty_sigma_scale = adaptive_penalty_sigma_scale
        self._classification_score_with_penalty = classification_score_with_penalty
        self._min_sigma = min_sigma

        self._fixed_window = None # Will be computed in start if needed

    def start(self):
        """Pre-computes scaling factors and the fixed window if needed."""
        # Scale factor for mapping relative box coordinates back to search region pixels
        # Assuming box format uses (height, width) scaling for (y, x) coordinates
        self._scale_factor = torch.tensor(
            (self._search_region_size[0], self._search_region_size[1]),
            dtype=torch.float, device=self._device)

        # Precompute fixed Hanning window if adaptive penalty is NOT used
        if self._enable_penalty and not self._use_adaptive_penalty:
            hann_h = torch.hann_window(self._response_map_size[0], periodic=False, device=self._device)
            hann_w = torch.hann_window(self._response_map_size[1], periodic=False, device=self._device)
            self._fixed_window = torch.outer(hann_h, hann_w).flatten() # Shape (H*W,)

        # Coordinate grid for adaptive window (only needs to be created once)
        if self._use_adaptive_penalty:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(self._response_map_size[0], dtype=torch.float, device=self._device),
                torch.arange(self._response_map_size[1], dtype=torch.float, device=self._device),
                indexing='ij' # Produces (H, W) grids
            )
            # Store as (1, H, W) for broadcasting with batch dim N later
            self._grid_y = grid_y.unsqueeze(0)
            self._grid_x = grid_x.unsqueeze(0)


    def stop(self):
        """Clears tensors."""
        # No need for explicit 'del', just reset references if desired for clarity
        self._scale_factor = None
        self._fixed_window = None
        if hasattr(self, '_grid_y'):
             self._grid_y = None
             self._grid_x = None

    def _create_adaptive_window(self, prior_box_batch: torch.Tensor) -> torch.Tensor:
        """
        Creates a batch of 2D Gaussian windows based on prior boxes.

        Args:
            prior_box_batch: Tensor of shape (N, 4) with prior boxes [cx, cy, w, h]
                             relative to the search region (coords 0-1 or pixel coords).
                             *Crucially, assumes pixel coordinates here matching
                             search_region_size*. If relative (0-1), uncomment scaling.

        Returns:
            Tensor of shape (N, H*W) containing the flattened Gaussian windows.
        """
        N = prior_box_batch.shape[0]
        H, W = self._response_map_size
        search_h, search_w = self._search_region_size

        # Extract center and size - ensure correct format [cx, cy, w, h]
        # Assuming prior_box is in PIXEL coordinates relative to search region
        center_x = prior_box_batch[:, 0:1] # Shape (N, 1)
        center_y = prior_box_batch[:, 1:2] # Shape (N, 1)
        box_w = prior_box_batch[:, 2:3]    # Shape (N, 1)
        box_h = prior_box_batch[:, 3:4]    # Shape (N, 1)

        # --- If prior_box is relative (0-1), uncomment this scaling ---
        # center_x = center_x * search_w
        # center_y = center_y * search_h
        # box_w = box_w * search_w
        # box_h = box_h * search_h
        # --- End scaling ---

        # Map center to response map grid coordinates (N, 1, 1) for broadcasting
        grid_center_x = (center_x / search_w * W).unsqueeze(2)
        grid_center_y = (center_y / search_h * H).unsqueeze(2)

        # Calculate sigma on the response map grid (N, 1, 1)
        # Use max with min_sigma to prevent issues with tiny/zero boxes
        sigma_x = torch.clamp(box_w / search_w * W * self._adaptive_penalty_sigma_scale, min=self._min_sigma).unsqueeze(2)
        sigma_y = torch.clamp(box_h / search_h * H * self._adaptive_penalty_sigma_scale, min=self._min_sigma).unsqueeze(2)

        # Calculate squared distance from center (broadcasts N,1,1 against 1,H,W -> N,H,W)
        dist_sq = ( (self._grid_x - grid_center_x) / sigma_x )**2 + \
                  ( (self._grid_y - grid_center_y) / sigma_y )**2

        # Calculate Gaussian (N, H, W)
        adaptive_window_2d = torch.exp(-0.5 * dist_sq)

        # Flatten to (N, H*W)
        return adaptive_window_2d.view(N, H * W)


    def __call__(self, output: dict, prior_box: list[np.ndarray]) -> dict:
        """
        Processes the raw tracker output, optionally applying a penalty window.

        Args:
            output: A dictionary containing:
                'score_map': Raw classification score map (N, H, W).
                'boxes': Raw bounding box predictions (N, H, W, 4). Assumed to be
                         relative coordinates (e.g., [cy, cx, h, w] or [cx, cy, w, h]
                         normalized 0-1 or similar, that will be scaled by
                         search_region_size). Format MUST match scaling logic below.
            prior_box: Optional tensor (N, 4) of prior bounding boxes [x1, y1, x2, y2]
                       in pixel coordinates relative to the search region. Required if
                       `use_adaptive_penalty` is True.

        Returns:
            A dictionary containing:
                'box': The predicted bounding box (N, 4) scaled to search_region_size.
                'confidence': The confidence score (N,) for the prediction.
        """
        # shape: (N, H, W), (N, H, W, 4)
        predicted_score_map = output['score_map'].detach().float().sigmoid()
        predicted_bbox = output['boxes'].detach().float()

        N, H, W = predicted_score_map.shape
        assert H == self._response_map_size[1] and W == self._response_map_size[0], "Score map size mismatch"

        flat_score_map = predicted_score_map.view(N, H * W) # (N, H*W)

        best_idx: torch.Tensor
        score_map_for_idx: torch.Tensor = flat_score_map

        if self._enable_penalty:
            window: Optional[torch.Tensor] = None
            if self._use_adaptive_penalty:
                prior_box = torch.from_numpy(np.stack(prior_box, axis=0))
                prior_box = bbox_xyxy_to_cxcywh_torch(prior_box)
                if prior_box is None:
                    raise ValueError("prior_box must be provided when use_adaptive_penalty is True")
                if prior_box.shape[0] != N:
                    raise ValueError(f"Batch size mismatch between prior_box ({prior_box.shape[0]}) and score_map ({N})")
                # Create window batch (N, H*W)
                window = self._create_adaptive_window(prior_box.to(self._device))
            else:
                # Use fixed window, expand batch dim (1, H*W -> N, H*W)
                if self._fixed_window is None:
                     raise RuntimeError("Fixed window not initialized. Call start() first.")
                window = self._fixed_window.expand(N, -1) # Use expand for efficiency

            if window is not None:
                # Apply penalty: Additive (original paper style) or Multiplicative
                # Additive: Boosts scores where window is high
                score_map_with_penalty = flat_score_map * (1.0 - self._window_penalty_ratio) + \
                                         window * self._window_penalty_ratio

                # --- OR ---
                # Multiplicative: Suppresses scores where window is low
                # penalized_window = 1.0 - self._window_penalty_ratio * (1.0 - window)
                # score_map_with_penalty = flat_score_map * penalized_window
                # --- Choose one penalty application method ---

                score_map_for_idx = score_map_with_penalty # Find max index in penalized map
                _, best_idx = torch.max(score_map_for_idx, 1, keepdim=True) # (N, 1)

                # Determine the confidence score to return
                if self._classification_score_with_penalty:
                     # Gather the score *after* penalty application
                    confidence_score = torch.gather(score_map_with_penalty, 1, best_idx)
                else:
                     # Gather the *original* score at the best location found using the penalty
                    confidence_score = torch.gather(flat_score_map, 1, best_idx)

            else: # Should not happen if _enable_penalty is True, but defensive check
                 confidence_score, best_idx = torch.max(flat_score_map, 1, keepdim=True)

        else:
            # No penalty: find max directly from original scores
            confidence_score, best_idx = torch.max(flat_score_map, 1, keepdim=True) # (N, 1)

        confidence_score = confidence_score.squeeze(1) # (N,)

        # Gather the corresponding bounding box
        flat_bbox = predicted_bbox.view(N, H * W, 4) # (N, H*W, 4)
        # Index needs shape (N, 1, 4) for gather
        idx_for_gather = best_idx.unsqueeze(-1).expand(-1, -1, 4) # (N, 1, 4)
        bounding_box = torch.gather(flat_bbox, 1, idx_for_gather).squeeze(1) # (N, 4)

        # Scale bounding box to search region size
        # IMPORTANT: This assumes the raw bbox output format is something like
        # [cy_rel, cx_rel, h_rel, w_rel] or [cx_rel, cy_rel, w_rel, h_rel] that aligns
        # with the self._scale_factor = [search_h, search_w]. Adjust if needed.
        # Example: If box is [cx, cy, w, h] relative, scaling might be:
        # scale_vec = self._scale_factor.repeat(1, 2).unsqueeze(0) # [1, 4] -> [search_w, search_h, search_w, search_h]
        # bounding_box = bounding_box * scale_vec
        # The original view-based scaling assumes a specific structure - double check it.
        # Let's assume the original code's scaling logic matches the network output format:
        # Assumes box is [..., 4] where the last dim corresponds to [y, x, h, w] style relative coords?
        # Or maybe [y1, x1, y2, x2]? Needs clarification based on the network's definition.
        # Keeping the original scaling logic:
        # Assumes gathered bounding_box is shape [N, 4] representing [[cy, cx], [h, w]] relative coords?
        # Scale factor is [search_h, search_w]
        # Commenting the assumed logic:
        # View as (N, 2 points, 2 coords), scale point coords, view back to (N, 4)
        scale_factor_view = self._scale_factor.view(1, 1, 2) # Shape (1, 1, 2)
        bounding_box_scaled = (bounding_box.view(N, 2, 2) * scale_factor_view).view(N, 4)
        # TODO: Verify this scaling logic matches the raw bbox output format precisely.

        return {'box': bounding_box_scaled, 'confidence': confidence_score}