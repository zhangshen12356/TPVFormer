from .tpv_head import TPVFormerHead
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
from mmseg.models import HEADS
from mmcv.cnn.bricks.transformer import build_positional_encoding, \
    build_transformer_layer_sequence
from mmcv.runner import force_fp32, auto_fp16

from .modules.cross_view_hybrid_attention import TPVCrossViewHybridAttention
from .modules.image_cross_attention import TPVMSDeformableAttention3D

@HEADS.register_module()
class TPVFormerHeadTRT(TPVFormerHead):
    def __init__(self, *args, **kwargs):
        super(TPVFormerHeadTRT, self).__init__(*args, **kwargs)
        
    def forward_trt(self, mlvl_feats, lidar2img, img_shape):
        bs = mlvl_feats[0].shape[0]
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device
        # tpv queries and pos embeds
        tpv_queries_hw = self.tpv_embedding_hw.weight.to(dtype)        # Embedding(100*100, 256)
        tpv_queries_zh = self.tpv_embedding_zh.weight.to(dtype)        # Embedding(8*100, 256)
        tpv_queries_wz = self.tpv_embedding_wz.weight.to(dtype)        # Embedding(100*8, 256)
        tpv_queries_hw = tpv_queries_hw.unsqueeze(0).repeat(bs, 1, 1)  # torch.Size([1, 10000, 256])
        tpv_queries_zh = tpv_queries_zh.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_wz = tpv_queries_wz.unsqueeze(0).repeat(bs, 1, 1)
        tpv_mask_hw = self.tpv_mask_hw.expand(bs, -1, -1)
        tpv_pos_hw = self.positional_encoding(tpv_mask_hw).to(dtype)
        tpv_pos_hw = tpv_pos_hw.flatten(2).transpose(1, 2)  # torch.Size([1, 10000, 256])
        
        # flatten image features of different scales
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2) # num_cam, bs, hw, c
            # 相机参数编码，？？？下面这步具体的作用是什么呢
            feat = feat + self.cams_embeds[:, None, None, :].to(dtype)  # self.cams_embeds[:, None, None, :]: torch.Size([6, 1, 1, 256])
            # 特征图尺度编码, ???下面这步具体的作用是什么
            feat = feat + self.level_embeds[None, None, lvl:lvl+1, :].to(dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        
        feat_flatten = torch.cat(feat_flatten, 2) # num_cam, bs, hw++, c  torch.Size([6, 1, 30825, 256])
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)  # torch.Size([4, 2])
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # tensor([    0, 23200, 29000, 30450], device='cuda:0')
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)  torch.Size([6, 30825, 1, 256])
        tpv_embed = self.encoder.forward_trt(
            [tpv_queries_hw, tpv_queries_zh, tpv_queries_wz],
            feat_flatten,
            feat_flatten,
            tpv_h=self.tpv_h, # 100
            tpv_w=self.tpv_w, # 100
            tpv_z=self.tpv_z, # 8
            tpv_pos=[tpv_pos_hw, None, None],
            # tpv_pos=[tpv_pos_hw],
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            lidar2img=lidar2img,
            img_shape=img_shape
        )
        return tpv_embed