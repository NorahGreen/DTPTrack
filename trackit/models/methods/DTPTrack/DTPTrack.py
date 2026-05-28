from typing import Tuple, Sequence, Optional
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from .modules.patch_embed import PatchEmbedNoSizeCheck
from .modules.head.mlp import MlpAnchorFreeHead_2
from trackit.models import ModelInputDataSelfDescriptionMixin
from .modules.frame_wise_causal_mlora.block import ChunkedCausalBlock_MoE


class DTPTrack(nn.Module, ModelInputDataSelfDescriptionMixin):
    def __init__(self, vit: VisionTransformer,
                 template_feat_size: Tuple[int, int],
                 search_region_feat_size: Tuple[int, int],
                 with_cls_token: bool, with_reg_token: bool,
                 enable_aux_output: bool, inference_mode: bool,
                 aux_output_multi_head: bool,
                 num_prior_tokens_per_frame: int = 1,
                 **kwargs):
        super().__init__()
        self.inference_mode = inference_mode
        self.z_size = template_feat_size
        self.x_size = search_region_feat_size
        self.inter_x_sizes = [template_feat_size, template_feat_size, template_feat_size]

        # --- Standard ViT ---
        assert isinstance(vit, VisionTransformer)
        self.embed_dim = vit.embed_dim
        self.patch_embed = PatchEmbedNoSizeCheck.build(vit.patch_embed)
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        for param in vit.parameters():
            param.requires_grad = False
        self.blocks = nn.ModuleList(ChunkedCausalBlock_MoE.copy_from_std_block(block, **kwargs)
                                    for block in vit.blocks)
        self.norms = nn.ModuleList(nn.LayerNorm(self.embed_dim, eps=1.e-5) for _ in range(len(self.inter_x_sizes) + 1))
        self.pos_drop = vit.pos_drop
        self.norm_pre = vit.norm_pre
        self.grid_size = vit.patch_embed.grid_size
        self.pos_embed = nn.Parameter(torch.empty(1, self.grid_size[0] * self.grid_size[1], self.embed_dim))
        self.pos_embed.data.copy_(vit.pos_embed.data[:, vit.num_prefix_tokens if not vit.no_embed_class else 0:, :])
        self.token_type_embed = nn.Parameter(torch.empty(2, self.embed_dim))
        nn.init.normal_(self.token_type_embed, std=1.e-6)
        self.search_region_token_type_embed = nn.ParameterList(nn.Parameter(torch.empty(1, self.embed_dim))
                                                               for _ in range(len(self.inter_x_sizes) + 1))
        for param in self.search_region_token_type_embed:
            nn.init.normal_(param, std=1.e-6)

        # --- TRC & TGS Gated Prior Logic ---
        self.num_prior_tokens_per_frame = num_prior_tokens_per_frame
        self.num_prior_provider_frames = len(self.inter_x_sizes) + 1
        total_prior_tokens = self.num_prior_provider_frames * self.num_prior_tokens_per_frame

        # TGS: base prior Token
        self.tgs_base_prior_token = nn.Parameter(torch.empty(1, total_prior_tokens, self.embed_dim))
        nn.init.normal_(self.tgs_base_prior_token, std=1.e-6)

        # TGS: modulator MLP (f_mod)
        self.tgs_modulator = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 4, self.embed_dim)
        )

        # TRC: Confidence Gated (f_gate)
        self.trc_gate = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 4, 1),
            nn.Sigmoid()
        )

        # TGS: Position and Type Embedding
        self.tgs_pos_embed = nn.Parameter(torch.empty(1, total_prior_tokens, self.embed_dim))
        nn.init.normal_(self.tgs_pos_embed, std=1.e-6)
        self.tgs_token_type_embed = nn.Parameter(torch.empty(1, self.embed_dim))
        nn.init.normal_(self.tgs_token_type_embed, std=1.e-6)

        # --- Head and Other Modules ---
        multi_head = enable_aux_output and aux_output_multi_head
        self.multi_head = multi_head
        if multi_head:
            self.head = nn.ModuleList(MlpAnchorFreeHead_2(self.embed_dim) for _ in range(len(self.inter_x_sizes) + 1))
        else:
            self.head = MlpAnchorFreeHead_2(self.embed_dim)

        self.cls_token = vit.cls_token if with_cls_token else None
        self.reg_token = vit.reg_token if with_reg_token else None

        self.section_sizes = self._get_token_sections()
        self.enable_aux_output = enable_aux_output

    def _get_target_summary(self, feat: torch.Tensor, mask: Optional[torch.Tensor]):
        if mask is None: return None
        mask_float = mask.flatten(1).unsqueeze(-1).float()
        summary_feat = (feat * mask_float).sum(dim=1) / (mask_float.sum(dim=1) + 1e-6)
        return summary_feat

    def forward(self, z_0: torch.Tensor, z_1: torch.Tensor, z_2: torch.Tensor, z_3: torch.Tensor,
                x_0: torch.Tensor,
                z_0_feat_mask: torch.Tensor,
                z_1_feat_mask: Optional[torch.Tensor] = None,
                z_2_feat_mask: Optional[torch.Tensor] = None,
                z_3_feat_mask: Optional[torch.Tensor] = None):

        z_feat = self._z_feat(z_0, z_0_feat_mask)
        x_feat_list = [
            self._inter_feat(z_1, 0),
            self._inter_feat(z_2, 1),
            self._inter_feat(z_3, 2),
            self._x_feat(x_0)
        ]
        B = z_feat.shape[0]

        all_template_feats = [z_feat] + x_feat_list[:-1]
        all_masks = [z_0_feat_mask, z_1_feat_mask, z_2_feat_mask, z_3_feat_mask]

        summaries = []
        for feat, mask in zip(all_template_feats, all_masks):
            summary = self._get_target_summary(feat, mask)
            if summary is None:
                summary = torch.zeros(B, self.embed_dim, device=feat.device, dtype=feat.dtype)
            summaries.append(summary)

        stacked_summaries = torch.stack(summaries, dim=1)

        # --- APPLY THE TRC (Temporal Reliability Calibrator) GATING MECHANISM ---
        trc_confidence_weights = self.trc_gate(stacked_summaries)

        trc_final_weights = torch.ones_like(trc_confidence_weights)

        if trc_final_weights.shape[1] > 1:
            trc_final_weights[:, 1:, :] = trc_confidence_weights[:, 1:, :]

        trc_calibrated_summaries = stacked_summaries * trc_final_weights

        # --- APPLY THE TGS (Temporal Guidance Synthesizer) MODULATION ---
        tgs_modulation_signals = self.tgs_modulator(trc_calibrated_summaries)

        tgs_signals_expanded = tgs_modulation_signals.repeat_interleave(self.num_prior_tokens_per_frame, dim=1)
        tgs_dynamic_prior_token = self.tgs_base_prior_token + tgs_signals_expanded
        tgs_dynamic_prior_token = tgs_dynamic_prior_token + self.tgs_pos_embed + self.tgs_token_type_embed

        all_feature_blocks = self._fusion(z_feat, x_feat_list, tgs_dynamic_prior_token)
        x_feature_blocks = all_feature_blocks[1:]

        feat_sizes = [self.z_size] + self.inter_x_sizes + [self.x_size]
        C = x_feature_blocks[0].shape[2]

        if self.enable_aux_output:
            x_feats_normalized = [self.norms[i](feat) for i, feat in enumerate(x_feature_blocks)]
            x_feat_sizes = feat_sizes[1:]
            feats = [feat.reshape(B, size[1], size[0], C) for feat, size in zip(x_feats_normalized, x_feat_sizes)]
            if self.multi_head:
                if self.inference_mode: return self.head[-1](feats[-1])
                return tuple(self.head[i](feat) for i, feat in enumerate(feats))
            else:
                return tuple(self.head(feat) for feat in feats)
        else:
            x_feat = self.norms[-1](x_feature_blocks[-1])
            H, W = self.x_size[1], self.x_size[0]
            x_feat = x_feat.reshape(B, H, W, C)
            return self.head(x_feat)

    def _z_feat(self, z: torch.Tensor, z_feat_mask: torch.Tensor):
        z = self.patch_embed(z)
        z_W, z_H = self.z_size
        z_pos_embed = self.pos_embed.view(1, self.grid_size[1], self.grid_size[0], self.embed_dim)[:, : z_H, : z_W, :].reshape(1, z_H * z_W, self.embed_dim)
        z = self.pos_drop(self.norm_pre(z + z_pos_embed))
        mask_indices = z_feat_mask.flatten(1)
        z = z + self.token_type_embed[mask_indices]
        return z

    def _inter_feat(self, x: torch.Tensor, index: int):
        x = self.patch_embed(x)
        N, L, _ = x.shape
        x_W, x_H = self.inter_x_sizes[index]
        x_pos_embed = self.pos_embed.view(1, self.grid_size[1], self.grid_size[0], self.embed_dim)[:, : x_H, : x_W, :].reshape(1, x_H * x_W, self.embed_dim)
        x = self.pos_drop(self.norm_pre(x + x_pos_embed))
        x = x + self.search_region_token_type_embed[index].unsqueeze(0).expand(N, L, -1)
        return x

    def _x_feat(self, x: torch.Tensor):
        x_feat = self.patch_embed(x)
        N, L, _ = x_feat.shape
        x_W, x_H = self.x_size
        x_pos_embed = self.pos_embed.view(1, self.grid_size[1], self.grid_size[0], self.embed_dim)[:, : x_H, : x_W, :].reshape(1, x_H * x_W, self.embed_dim)
        x_feat = self.pos_drop(self.norm_pre(x_feat + x_pos_embed))
        x_feat = x_feat + self.search_region_token_type_embed[-1].unsqueeze(0).expand(N, L, -1)
        return x_feat

    def _fusion(self, z_feat: torch.Tensor, x_feat_list: Sequence[torch.Tensor], tgs_dynamic_prior_token: torch.Tensor):
        B = z_feat.shape[0]
        tokens = self._get_prefix_tokens(B)
        tokens.append(tgs_dynamic_prior_token)
        tokens.append(z_feat)
        tokens.extend(x_feat_list)
        fusion_feat = torch.cat(tokens, dim=1)
        chunk_sizes = self.section_sizes
        past_kv_enable_bits = [False] + [True] * len(x_feat_list)
        expert_idx = [0] * len(chunk_sizes)
        for block in self.blocks:
            fusion_feat = block(fusion_feat, expert_idx=expert_idx, chunk_sizes=chunk_sizes, past_kv_enable_bits=past_kv_enable_bits)
        prefix_len = self._get_number_of_prefix_tokens() + tgs_dynamic_prior_token.shape[1]
        split_sizes = [prefix_len] + [self.z_size[0] * self.z_size[1]] + [s[0] * s[1] for s in self.inter_x_sizes] + \
            [self.x_size[0] * self.x_size[1]]
        all_tokens = torch.split(fusion_feat, split_sizes, dim=1)
        return all_tokens[1:]

    def _get_token_sections(self):
        prefix_tokens = (
                    self._get_number_of_prefix_tokens() + self.num_prior_provider_frames * self.num_prior_tokens_per_frame)
        token_sections = [prefix_tokens + self.z_size[0] * self.z_size[1]]
        token_sections.extend([s[0] * s[1] for s in self.inter_x_sizes])
        token_sections.append(self.x_size[0] * self.x_size[1])
        return token_sections

    def _get_prefix_tokens(self, batch_size: int):
        tokens = []
        if self.cls_token is not None: tokens.append(self.cls_token.repeat(batch_size, 1, 1))
        if self.reg_token is not None: tokens.append(self.reg_token.repeat(batch_size, 1, 1))
        return tokens

    def _get_number_of_prefix_tokens(self):
        num = 0
        if self.cls_token is not None: num += self.cls_token.shape[1]
        if self.reg_token is not None: num += self.reg_token.shape[1]
        return num

    def get_sample_data(self, batch_size: int, device: torch.device, dtype: torch.dtype,
                        auto_mixed_precision_dtype: Optional[torch.dtype]):
        z_feat_size = self.z_size
        x_feat_size = self.x_size
        z_size = (z_feat_size[0] * self.patch_embed.patch_size[1], z_feat_size[1] * self.patch_embed.patch_size[0])
        x_size = (x_feat_size[0] * self.patch_embed.patch_size[1], x_feat_size[1] * self.patch_embed.patch_size[0])
        inter_x_sizes = tuple(
            (s[0] * self.patch_embed.patch_size[1], s[1] * self.patch_embed.patch_size[0]) for s in self.inter_x_sizes)
        args = [torch.full((batch_size, 3, z_size[1], z_size[0]), 0.5, device=device, dtype=dtype)]
        for i in range(len(self.inter_x_sizes)):
            args.append(
                torch.full((batch_size, 3, inter_x_sizes[i][1], inter_x_sizes[i][0]), 0.5, device=device, dtype=dtype))
        args.append(torch.full((batch_size, 3, x_size[1], x_size[0]), 0.5, device=device, dtype=dtype))
        args.extend([
            torch.full((batch_size, z_feat_size[1], z_feat_size[0]), 1, dtype=torch.long, device=device),
            torch.full((batch_size, self.inter_x_sizes[0][1], self.inter_x_sizes[0][0]), 1, dtype=torch.long,
                       device=device),
            torch.full((batch_size, self.inter_x_sizes[1][1], self.inter_x_sizes[1][0]), 1, dtype=torch.long,
                       device=device),
            torch.full((batch_size, self.inter_x_sizes[2][1], self.inter_x_sizes[2][0]), 1, dtype=torch.long,
                       device=device)
        ])
        return tuple(args)
