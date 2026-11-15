"""Binary vit_t architecture from the pinned SegmentAnyBone build_sam.py.

Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
SegmentAnyBone adaptations distributed under CC BY-NC 4.0; see LICENSE and
PROVENANCE.json. Original SAM and TinyViT notices are retained alongside it.
"""
from types import SimpleNamespace

import torch
from torch import nn

from .tiny_vit_sam import TinyViT
from .prompt_encoder import PromptEncoder, attention_fusion
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer


def build_bone_model():
    """Construct every checkpoint parameter, including unused depth fusion.

    This is the trained vit_t binary configuration, including the encoder and
    decoder adapters enabled in demo.ipynb. The caller must strictly load weights BEFORE to(device).eval(),
    which builds TinyViT's nonpersistent learned attention-bias caches.
    """
    args = SimpleNamespace(if_encoder_adapter=True, encoder_adapter_depths=(0, 1, 10, 11),
                           decoder_adapt_depth=2, if_mask_decoder_adapter=True)
    model = nn.Module()
    model.image_encoder = TinyViT(
        args, img_size=1024, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 160, 320], depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10], window_sizes=[7, 7, 14, 7],
        mlp_ratio=4., drop_rate=0., drop_path_rate=0., use_checkpoint=False,
        mbconv_expand_ratio=4., local_conv_size=3, layer_lr_decay=.8)
    model.prompt_encoder = PromptEncoder(
        embed_dim=256, image_embedding_size=(64, 64),
        input_image_size=(1024, 1024), mask_in_chans=16)
    model.mask_decoder = MaskDecoder(
        num_multimask_outputs=2,
        transformer=TwoWayTransformer(args, depth=2, embedding_dim=256,
                                      mlp_dim=2048, num_heads=8),
        transformer_dim=256, iou_head_depth=3, iou_head_hidden_dim=256)
    model.attention_fusion = attention_fusion()
    model.register_buffer('pixel_mean', torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), persistent=False)
    model.register_buffer('pixel_std', torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), persistent=False)
    return model
