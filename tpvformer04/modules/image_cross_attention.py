
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule
from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class TPVImageCrossAttention(BaseModule):
    """Image cross attention module used in TPVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 tpv_h=None,
                 tpv_w=None,
                 tpv_z=None,
                 **kwargs
                 ):
        super().__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                # residual=None,
                reference_points_cams=None,
                tpv_masks=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_cam, H*W++, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_cam, H*W++, bs, embed_dims)`.
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        if key is None:
            key = query
        if value is None:
            value = key
        # if residual is None:
        #     inp_residual = query
        inp_residual = query

        bs, num_query, _ = query.size()

        queries = torch.split(
            query, [self.tpv_h*self.tpv_w, self.tpv_z*self.tpv_h, self.tpv_w*self.tpv_z], dim=1)
        # if residual is None:
        #     slots = [torch.zeros_like(q) for q in queries]
        slots = [torch.zeros_like(q) for q in queries]
        indexeses = []
        max_lens = []
        queries_rebatches = []
        reference_points_rebatches = []
        # import pdb; pdb.set_trace()
        # len(tpv_masks)=3  tpv_masks[0]:torch.Size([6, 1, 10000, 4])  tpv_masks[1]:torch.Size([6, 1, 800, 32])  tpv_masks[2]:torch.Size([6, 1, 800, 32])
        for tpv_idx, tpv_mask in enumerate(tpv_masks):
            indexes = []
            for _, mask_per_img in enumerate(tpv_mask):
                # change torch.Size([1546])
                index_query_per_img = mask_per_img[0].sum(
                    -1).nonzero().squeeze(-1)
                indexes.append(index_query_per_img)
            if torch.onnx.is_in_onnx_export():
                max_len = torch.max(torch.stack([each.shape[0] for each in indexes]))
            else:
                max_len = max([each.shape[0] for each in indexes])
            max_lens.append(max_len)
            indexeses.append(indexes)

            reference_points_cam = reference_points_cams[tpv_idx]
            D = reference_points_cam.size(3)

            queries_rebatch = queries[tpv_idx].new_zeros(
                [bs * self.num_cams, max_len, self.embed_dims])  # [1*6, 2380, 256]
            reference_points_rebatch = reference_points_cam.new_zeros(
                [bs * self.num_cams, max_len, D, 2])
            for i, reference_points_per_img in enumerate(reference_points_cam):
                for j in range(bs):
                    index_query_per_img = indexes[i]
                    queries_rebatch[j * self.num_cams + i, :index_query_per_img.shape[0]] = queries[tpv_idx][j, index_query_per_img]
                    reference_points_rebatch[j * self.num_cams + i, :index_query_per_img.shape[0]] = reference_points_per_img[j, index_query_per_img]
            queries_rebatches.append(queries_rebatch)
            reference_points_rebatches.append(reference_points_rebatch)

        num_cams, l, bs, embed_dims = key.shape
        key = key.permute(0, 2, 1, 3).view(
            self.num_cams * bs, l, self.embed_dims)
        value = value.permute(0, 2, 1, 3).view(
            self.num_cams * bs, l, self.embed_dims)
        queries = self.deformable_attention(
            query=queries_rebatches, key=key, value=value,
            reference_points=reference_points_rebatches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index)

        for tpv_idx, indexes in enumerate(indexeses):
            for i, index_query_per_img in enumerate(indexes):
                for j in range(bs):
                    slots[tpv_idx][j, index_query_per_img] += queries[tpv_idx][j * self.num_cams + i, :index_query_per_img.shape[0]]
            count = tpv_masks[tpv_idx].sum(-1) > 0
            count = count.permute(1, 2, 0).sum(-1)
            count = torch.clamp(count, min=1.0)
            slots[tpv_idx] = slots[tpv_idx] / count[..., None]
        # --------------------------------------------------------------
        # slots_hw = slots[0]
        # slots_hz = slots[1]
        # slots_zw = slots[2]
        
        # indexex_hw = indexeses[0]
        # indexex_hz= indexeses[1]
        # indexex_zw = indexeses[2]
        # for i, index_query_per_img in enumerate(indexex_hw):
        #     for j in range(bs):
        #         slots_hw[j, index_query_per_img] += queries[0][j * self.num_cams + i, :index_query_per_img.shape[0]] 
                
        # for i, index_query_per_img in enumerate(indexex_hz):
        #     for j in range(bs):
        #         slots_hz[j, index_query_per_img] += queries[1][j * self.num_cams + i, :index_query_per_img.shape[0]] 
                
        # for i, index_query_per_img in enumerate(indexex_zw):
        #     for j in range(bs):
        #         slots_zw[j, index_query_per_img] += queries[2][j * self.num_cams + i, :index_query_per_img.shape[0]] 
        # # import pdb; pdb.set_trace()
        # count_hw = tpv_masks[0].sum(-1) > 0
        # count_hz = tpv_masks[1].sum(-1) > 0
        # count_zw = tpv_masks[2].sum(-1) > 0
        
        # count_hw = count_hw.permute(1, 2, 0)
        # count_hw = count_hw.sum(-1)
        # count_hz = count_hz.permute(1, 2, 0)
        # count_hz = count_hz.sum(-1)
        # count_zw = count_zw.permute(1, 2, 0)
        # count_zw = count_zw.sum(-1)
        
        # count_hw = torch.clamp(count_hw, min=1.0)
        # count_hz = torch.clamp(count_hz, min=1.0)
        # count_zw = torch.clamp(count_zw, min=1.0)
        # # import pdb; pdb.set_trace()
        # count_hw = torch.reciprocal(count_hw)
        # count_hz = torch.reciprocal(count_hz)
        # count_zw = torch.reciprocal(count_zw)
        # count_hw = torch.reciprocal(count_hw)[..., None].repeat(1, 1, embed_dims)
        # count_hz = torch.reciprocal(count_hz)[..., None].repeat(1, 1, embed_dims)
        # count_zw = torch.reciprocal(count_zw)[..., None].repeat(1, 1, embed_dims)
        # # import pdb; pdb.set_trace()
        # slots_hw = slots_hw * count_hw
        # slots_hz = slots_hz * count_hz
        # slots_zw = slots_zw * count_zw
        # # for i in range(slots_hw.shape[2]):
        # #     slots_hw[:, :, i] = slots_hw[:, :, i] * count_hw
        # # for i in range(slots_hz.shape[2]):
        # #     slots_hz[:, :, i] = slots_hz[:, :, i] * count_hz
        # # for i in range(slots_zw.shape[2]):
        # #     slots_zw[:, :, i] = slots_zw[:, :, i] * count_zw
        
        
        # # slots[tpv_idx] = slots[tpv_idx] / count[..., None]
        # slots = torch.cat([slots_hw, slots_hz, slots_zw], dim=1)
        # --------------------------------------------------------------
        slots = torch.cat(slots, dim=1)
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class TPVMSDeformableAttention3D(BaseModule):
    """An attention module used in TPVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=[8, 64, 64],
                 num_z_anchors=[4, 32, 32],
                 pc_range=None,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None,
                 floor_sampling_offset=True,
                 tpv_h=None,
                 tpv_w=None,
                 tpv_z=None,
                 ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_z_anchors = num_z_anchors
        self.base_num_points = num_points[0]
        self.base_z_anchors = num_z_anchors[0]
        self.points_multiplier = [
            points // self.base_z_anchors for points in num_z_anchors]
        self.pc_range = pc_range
        self.tpv_h, self.tpv_w, self.tpv_z = tpv_h, tpv_w, tpv_z
        self.sampling_offsets = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i] * 2) for i in range(3)
        ])
        self.floor_sampling_offset = floor_sampling_offset
        self.attention_weights = nn.ModuleList([
            nn.Linear(embed_dims, num_heads * num_levels * num_points[i]) for i in range(3)
        ])
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        for i in range(3):
            constant_init(self.sampling_offsets[i], 0.)
            thetas = torch.arange(
                self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
                self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points[i], 1)
            grid_init = grid_init.reshape(
                self.num_heads, self.num_levels, self.num_z_anchors[i], -1, 2)
            for j in range(self.num_points[i] // self.num_z_anchors[i]):
                grid_init[:, :, :, j, :] *= j + 1

            self.sampling_offsets[i].bias.data = grid_init.view(-1)
            constant_init(self.attention_weights[i], val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def get_sampling_offsets_and_attention(self, queries):
        offsets = []
        attns = []
        for i, (query, fc, attn) in enumerate(zip(queries, self.sampling_offsets, self.attention_weights)):
            bs, l, d = query.shape
            offset = fc(query)
            # import pdb; pdb.set_trace()
            dim1, dim2, dim3 = offset.shape
            offset = offset.reshape(int(dim1), int(dim2), self.num_heads, self.num_levels, self.points_multiplier[i], int(dim3/self.num_heads/self.num_levels/self.points_multiplier[i]/2), 2)
            # offset = offset.reshape(bs, l, self.num_heads,
            #                            self.num_levels, self.points_multiplier[i], -1, 2)
            offset = offset.permute(0, 1, 4, 2, 3, 5, 6).flatten(1, 2)
            offsets.append(offset)
            attention = attn(query)
            dim1_1, dim2_1, dim3_1 = attention.shape
            attention = attention.reshape(int(dim1_1), int(dim2_1), self.num_heads, int(dim3_1/self.num_heads))
            # attention = attention.reshape(bs, l, self.num_heads, -1)
            attention = attention.softmax(-1)
            dim1_2, dim2_2, dim3_2, dim4_2 = attention.shape
            attention = attention.view(
                int(dim1_2), int(dim2_2), int(dim3_2), self.num_levels, self.points_multiplier[i], int(dim4_2/self.num_levels/self.points_multiplier[i]))
            # attention = attention.view(
            #     bs, l, self.num_heads, self.num_levels, self.points_multiplier[i], -1)
            attention = attention.permute(0, 1, 4, 2, 3, 5).flatten(1, 2)
            attns.append(attention)

        offsets = torch.cat(offsets, dim=1)
        attns = torch.cat(attns, dim=1)
        return offsets, attns

    def reshape_reference_points(self, reference_points):
        reference_point_list = []
        for i, reference_point in enumerate(reference_points):
            bs, l, z_anchors, dim_last = reference_point.shape
            reference_point = reference_point.reshape(
                int(bs), int(l), self.points_multiplier[i], int(z_anchors/self.points_multiplier[i]), int(dim_last))
            reference_point = reference_point.flatten(1, 2)
            reference_point_list.append(reference_point)
        return torch.cat(reference_point_list, dim=1)

    def reshape_output(self, output, lens):
        bs, _, d = output.shape
        outputs = torch.split(output, [lens[0]*self.points_multiplier[0], lens[1]
                              * self.points_multiplier[1], lens[2]*self.points_multiplier[2]], dim=1)

        outputs = [o.reshape(int(bs), -1, self.points_multiplier[i], int(d)).sum(dim=2)
                   for i, o in enumerate(outputs)]
        return outputs

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = [q.permute(1, 0, 2) for q in query]
            value = value.permute(1, 0, 2)

        # bs, num_query, _ = query.shape
        query_lens = [q.shape[1] for q in query]
        bs, num_value, _ = value.shape
        # assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(int(bs), int(num_value), self.num_heads, -1)

        sampling_offsets, attention_weights = self.get_sampling_offsets_and_attention(query)

        reference_points = self.reshape_reference_points(reference_points)
        # import pdb; pdb.set_trace()
        # num_cam, _, _, coordinate = reference_points.shape
        
        if int(reference_points.shape[-1]) == 2:
            """
            For each TPV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each TPV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, :, None, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            
            sampling_offsets = sampling_offsets.view(
                int(bs), int(num_query), int(num_heads), int(num_levels), int(num_Z_anchors), int(num_all_points / num_Z_anchors), int(xy))
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                int(bs), int(num_query), int(num_heads), int(num_levels), int(num_all_points), int(xy))
            
            if self.floor_sampling_offset:
                sampling_locations = sampling_locations - torch.floor(sampling_locations)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points

        if torch.cuda.is_available() and value.is_cuda and not torch.onnx.is_in_onnx_export():
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        output = self.reshape_output(output, query_lens)
        if not self.batch_first:
            output = [o.permute(1, 0, 2) for o in output]

        return output