# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

# Architectures from Facebook are adjusted such that Crowd Counting can be performed.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from models.DeiT.timm_functional.timm_functional import VisionTransformer_functional, _cfg
from timm.models.registry import register_model

# __all__ = [
#     'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224', 'deit_tiny_cnn_patch16_224',
#     'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
#     'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
#     'deit_base_distilled_patch16_384',
# ]

__all__ = ['deit_tiny_distilled_patch16_224_functional', 'deit_small_distilled_patch16_224_functional']


# ======================================================================================================= #
#                                        MODULES TO DO REGRESSION                                         #
# ======================================================================================================= #

class DeiTRegressionHead_functional(nn.Module):
    def __init__(self, crop_size):
        super().__init__()

        self.crop_size = crop_size

    def forward(self, pre_den, weights):
        pre_den = F.linear(pre_den, weights['regression_head.regression_head.lin_scaler.0.weight'],
                           bias=weights['regression_head.regression_head.lin_scaler.0.bias'])
        pre_den = F.relu(pre_den)
        pre_den = F.linear(pre_den, weights['regression_head.regression_head.lin_scaler.2.weight'],
                           bias=weights['regression_head.regression_head.lin_scaler.2.bias'])

        pre_den = pre_den.transpose(1, 2)
        den = F.fold(pre_den, (self.crop_size, self.crop_size), kernel_size=16, stride=16)

        return den


class RegressionTransformer_functional(VisionTransformer_functional):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.regression_head = DeiTRegressionHead_functional(kwargs['img_size'])

    def forward(self, x, weights, training):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # Adjusted to do Crowd Counting regression

        batch_size = x.shape[0]
        x = self.patch_embed(x, weights)

        # This token has been stolen by a lot of people now
        cls_tokens = weights['cls_token'].expand(batch_size, -1, -1)
        dist_token = weights['dist_token'].expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + weights['pos_embed']
        x = F.dropout(x, self.drop_rate, training=training)

        for blk in self.blocks:
            x = blk(x, weights, training)

        # pre_count = x[:, 0]
        pre_den = x[:, 2:]

        den = self.regression_head(pre_den, weights)

        return den



#
# class DistilledRegressionTransformer(VisionTransformer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#         num_patches = self.patch_embed.num_patches
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
#         self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
#
#         trunc_normal_(self.dist_token, std=.02)
#         trunc_normal_(self.pos_embed, std=.02)
#
#         self.regression_head = DeiTRegressionHead(kwargs['img_size'], kwargs['embed_dim'], self._init_weights)
#
#         self.head_dist.apply(self._init_weights)
#
#     def forward(self, x):
#         # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#         # with slight modifications to add the dist_token
#         # and now also Crowd Counting regression
#         B = x.shape[0]
#         x = self.patch_embed(x)
#
#         # This token has been stolen by a lot of people
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         dist_token = self.dist_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, dist_token, x), dim=1)
#
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         pre_count = x[:, 0]
#         pre_den = x[:, 2:]
#
#         den, count = self.regression_head(pre_count, pre_den)
#
#         return den, count


# ======================================================================================================= #
#                                               TINY MODEL                                                #
# ======================================================================================================= #
@register_model
def deit_tiny_distilled_patch16_224_functional(init_path=None, pretrained=False, **kwargs):
    model = RegressionTransformer_functional(
        img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    model.crop_size = 224
    model.n_patches = 14

    return model
#
#
# @register_model
# def deit_tiny_distilled_patch16_224(init_path=None, pretrained=False, **kwargs):
#     model = DistilledRegressionTransformer(
#         img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     model.crop_size = 224
#     model.n_patches = 14
#
#     if init_path:
#         model = init_model_state(model, init_path)
#
#     return model
#
#
# # ======================================================================================================= #
# #                                               SMALL MODEL                                               #
# # ======================================================================================================= #

# def deit_tiny_distilled_patch16_224_functional(init_path=None, pretrained=False, **kwargs):
#     model = RegressionTransformer_functional(
#         img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     model.crop_size = 224
#     model.n_patches = 14
#
#     return model
# @register_model
# def deit_small_patch16_224(init_path=None, pretrained=False, **kwargs):
#     model = RegressionTransformer(
#         img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     model.crop_size = 224
#     model.n_patches = 14
#
#     if init_path:
#         model = init_model_state(model, init_path)
#
#     return model
#
#
@register_model
def deit_small_distilled_patch16_224_functional(init_path=None, pretrained=False, **kwargs):
    model = RegressionTransformer_functional(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    model.crop_size = 224
    model.n_patches = 14

    return model
#
#
# # ======================================================================================================= #
# #                                               BASE MODEL                                                #
# # ======================================================================================================= #
#
# @register_model
# def deit_base_patch16_224(init_path, pretrained=False, **kwargs):
#     model = RegressionTransformer(
#         img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     model.crop_size = 224
#     model.n_patches = 14
#
#     if init_path:
#         model = init_model_state(model, init_path)
#
#     return model
#
#
# @register_model
# def deit_base_distilled_patch16_224(init_path, pretrained=False, **kwargs):
#     model = DistilledRegressionTransformer(
#         img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     model.crop_size = 224
#     model.n_patches = 14
#
#     if init_path:
#         model = init_model_state(model, init_path)
#
#     return model
#
# @register_model
# def deit_base_patch16_384(init_path=None, pretrained=False, **kwargs):
#     model = RegressionTransformer(
#         img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     model.crop_size = 384
#     model.n_patches = 24
#
#     if init_path:
#         model = init_model_state(model, init_path)
#
#     return model
#
#
# @register_model
# def deit_base_distilled_patch16_384(init_path=None, pretrained=False, **kwargs):
#     model = DistilledRegressionTransformer(
#         img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     model.crop_size = 384
#     model.n_patches = 24
#
#     if init_path:
#         model = init_model_state(model, init_path)
#
#     return model
#
#
# # ======================================================================================================= #
# #                                             UTIL FUNCTIONS                                              #
# # ======================================================================================================= #
#
# def init_model_state(model, init_path):
#     if init_path.startswith('https'):
#         checkpoint = torch.hub.load_state_dict_from_url(
#             init_path, map_location='cpu', check_hash=True)
#     else:
#         checkpoint = torch.load(init_path, map_location='cpu')
#     pretrained_state = checkpoint['model']
#     modified_model_state = model.state_dict()
#     # With this, we are able to load the pretrained modules while ignoring the new regression modules.
#     for key in pretrained_state.keys():
#         modified_model_state[key] = pretrained_state[key]
#     model.load_state_dict(modified_model_state)
#
#     return model
