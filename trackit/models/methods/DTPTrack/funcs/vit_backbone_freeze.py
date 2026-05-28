from itertools import chain
import torch.nn as nn


def freeze_vit_backbone_(model: nn.Module) -> nn.Module:
    """
    Freeze all backbone weights of a ViT‑style model *in‑place*.

    Notes
    -----
    * The routine assumes the model exposes attributes
      `patch_embed`, `blocks`, `pos_embed`, and `norm`.
    * Call *before* optimizer creation.
    """
    to_freeze = chain(
        model.patch_embed.parameters(),
        model.blocks.parameters(),
        (model.pos_embed,)
    )
    if hasattr(model, 'norm'):
        to_freeze = chain(to_freeze, model.norm.parameters())
    for p in to_freeze:
        p.requires_grad = False
    return model


def freeze_radio_backbone_(model: nn.Module) -> nn.Module:
    to_freeze = chain(model.patch_generator.parameters(), model.blocks.parameters()
    )
    for p in to_freeze:
        p.requires_grad = False
    return model


def freeze_loratv2_(self: nn.Module, freeze_norm: bool = True, freeze_head: bool = True):
    for param in chain(self.patch_embed.parameters(), self.blocks.parameters(), (self.pos_embed,)):
        param.requires_grad = False
    if freeze_norm:
        for param in self.norm.parameters():
            param.requires_grad = False
    if freeze_head:
        for param in self.head.parameters():
            param.requires_grad = False

    if hasattr(self, 'cls_token') and self.cls_token is not None:
        self.cls_token.requires_grad = False
    if hasattr(self, 'cls_token_pos_embed') and self.cls_token_pos_embed is not None:
        self.cls_token_pos_embed.requires_grad = False
    if hasattr(self, 'reg_token') and self.reg_token is not None:
        self.reg_token.requires_grad = False
    if hasattr(self, 'reg_token_pos_embed') and self.reg_token_pos_embed is not None:
        self.reg_token_pos_embed.requires_grad = False

