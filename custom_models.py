# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class RegressionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.regression_head = nn.ModuleDict({
            'global_counter': nn.Sequential(
                nn.Linear(kwargs['embed_dim'], 512),
                nn.ReLu(),
                nn.Linear(512, 1)
            ),
            'lin_scaler': nn.Sequential(
                nn.Linear(kwargs['embed_dim'], 512),
                nn.ReLu(),
                nn.Linear(512, 256)
            ),
            'folder': nn.Fold((kwargs['img_size'], kwargs['img_size']), kernel_size=16, stride=16)
        })

        self.head_dist.apply(self._init_weights)
        self.regression_head['global_counter'].apply(self._init_weights)
        self.regression_head['lin_scaler'].apply(self._init_weights)

    def forward(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # Adjusted to do Crowd Counting regression

        batch_size = x.shape[0]
        x = self.patch_embed(x)

        # This token has been stolen by a lot of people now
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        pre_count = x[:, 0]
        pre_den = x[:, 1:]

        count = self.regression_head['global_counter'](pre_count)

        pre_den = self.regression_head['lin_scaler'](pre_den)
        pre_den = pre_den.transpose(1, 2)
        den = self.regression_head['folder'](pre_den)

        return den, count


class DistilledRegressionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

        self.regression_head = nn.ModuleDict({
            'global_counter': nn.Sequential(
                nn.Linear(kwargs['embed_dim'], 512),
                nn.ReLu(),
                nn.Linear(512, 1)
            ),
            'lin_scaler': nn.Sequential(
                nn.Linear(kwargs['embed_dim'], 512),
                nn.ReLu(),
                nn.Linear(512, 256)
            ),
            'folder': nn.Fold((kwargs['img_size'], kwargs['img_size']), kernel_size=16, stride=16)
        })

        self.head_dist.apply(self._init_weights)
        self.regression_head['global_counter'].apply(self._init_weights)
        self.regression_head['lin_scaler'].apply(self._init_weights)


    def forward(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        # and now also Crowd Counting regression
        B = x.shape[0]
        x = self.patch_embed(x)

        # This token has been stolen by a lot of people
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        pre_count = x[:, 0]
        pre_den = x[:, 2:]

        count = self.regression_head['global_counter'](pre_count)

        pre_den = self.regression_head['lin_scaler'](pre_den)
        pre_den = pre_den.transpose(1, 2)
        den = self.regression_head['folder'](pre_den)

        return den, count


def init_model_state(model, init_path):
    if init_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            init_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(init_path, map_location='cpu')
    pretrained_state = checkpoint['model']
    modified_model_state = model.state_dict()
    # With this, we are able to load the pretrained modules while ignoring the new regression modules.
    for key in pretrained_state.keys():
        modified_model_state[key] = pretrained_state[key]
    model.load_state_dict(modified_model_state)

    return model


@register_model
def deit_tiny_distilled_patch16_224(init_path=None, pretrained=False, **kwargs):
    model = DistilledRegressionTransformer(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model.patch_size = 224
    model.n_patches = 14

    if init_path:
        model = init_model_state(model, init_path)

    return model


@register_model
def deit_small_distilled_patch16_224(init_path=None, pretrained=False, **kwargs):
    model = DistilledRegressionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model.patch_size = 224
    model.n_patches = 14

    if init_path:
        model = init_model_state(model, init_path)

    return model


@register_model
def deit_base_distilled_patch16_224(init_path, pretrained=False, **kwargs):
    model = DistilledRegressionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model.patch_size = 224
    model.n_patches = 14

    if init_path:
        model = init_model_state(model, init_path)

    return model


@register_model
def deit_base_distilled_patch16_384(init_path=None, pretrained=False, **kwargs):
    model = DistilledRegressionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    model.patch_size = 384
    model.n_patches = 24

    if init_path:
        model = init_model_state(model, init_path)

    return model
