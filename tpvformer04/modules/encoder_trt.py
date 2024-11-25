from .encoder import TPVFormerEncoder

from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TPVFormerEncoderTRT(TPVFormerEncoder):
    def __init__(self, *args,  **kwargs):
        super(TPVFormerEncoderTRT, self).__init__(*args, **kwargs)
        # assert self.num_points_in_pillar[1] == self.num_points_in_pillar[2] and self.num_points_in_pillar[1] % self.num_points_in_pillar[0] == 0
        # ref_3d_hw = self.get_reference_points_trt(self.tpv_h, self.tpv_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar[0], '3d', device='cpu')

        # ref_3d_zh = self.get_reference_points_trt(self.tpv_z, self.tpv_h, self.pc_range[3]-self.pc_range[0], self.num_points_in_pillar[1], '3d', device='cpu')
        # ref_3d_zh = ref_3d_zh.permute(3, 0, 1, 2)[[2, 0, 1]]
        # ref_3d_zh = ref_3d_zh.permute(1, 2, 3, 0)

        # ref_3d_wz = self.get_reference_points_trt(self.tpv_w, self.tpv_z, self.pc_range[4]-self.pc_range[1], self.num_points_in_pillar[2], '3d', device='cpu')
        # ref_3d_wz = ref_3d_wz.permute(3, 0, 1, 2)[[1, 2, 0]]
        # ref_3d_wz = ref_3d_wz.permute(1, 2, 3, 0)
        # self.register_buffer('ref_3d_hw', ref_3d_hw)
        # self.register_buffer('ref_3d_zh', ref_3d_zh)
        # self.register_buffer('ref_3d_wz', ref_3d_wz)

        # ref_2d_hw = self.get_reference_points_trt(self.tpv_h, self.tpv_w, dim='2d', bs=1, device='cpu')
        # ref_2d_zh = self.get_reference_points_trt(self.tpv_z, self.tpv_h, dim='2d', bs=1, device='cpu')
        # ref_2d_wz = self.get_reference_points_trt(self.tpv_w, self.tpv_z, dim='2d', bs=1, device='cpu')
        # self.register_buffer('ref_2d_hw', ref_2d_hw)
        # self.register_buffer('ref_2d_zh', ref_2d_zh)
        # self.register_buffer('ref_2d_wz', ref_2d_wz)

    # @staticmethod
    # def get_reference_points_trt(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
    #     """Get the reference points used in spatial cross-attn and self-attn.
    #     Args:
    #         H, W: spatial shape of tpv plane.
    #         Z: hight of pillar.
    #         D: sample D points uniformly from each pillar.
    #         device (obj:`device`): The device where
    #             reference_points should be.
    #     Returns:
    #         Tensor: reference points used in decoder, has \
    #             shape (bs, num_keys, num_levels, 2).
    #     """

    #     # reference points in 3D space, used in spatial cross-attention (SCA)
    #     if dim == '3d':
    #         zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
    #                             device=device).view(-1, 1, 1).repeat(num_points_in_pillar, H, W) / Z
    #         xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
    #                             device=device).view(1, 1, -1).repeat(num_points_in_pillar, H, W) / W
    #         ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
    #                             device=device).view(1, -1, 1).repeat(num_points_in_pillar, H, W) / H
    #         ref_3d = torch.stack((xs, ys, zs), -1)
    #         ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    #         ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
    #         return ref_3d

    #     # reference points on 2D plane, used in temporal self-attention (TSA).
    #     elif dim == '2d':
    #         ref_y, ref_x = torch.meshgrid(
    #             torch.linspace(
    #                 0.5, H - 0.5, H, dtype=dtype, device=device),
    #             torch.linspace(
    #                 0.5, W - 0.5, W, dtype=dtype, device=device)
    #         )
    #         ref_y = ref_y.reshape(-1)[None] / H
    #         ref_x = ref_x.reshape(-1)[None] / W
    #         ref_2d = torch.stack((ref_x, ref_y), -1)
    #         ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
    #         return ref_2d

    def point_sampling_trt(self, reference_points, pc_range, lidar2img, img_shape):
        reference_points = reference_points.clone()
        # 变换到点云的范围内，这也是为何get_reference_points中会/W, /H, /Z，先归一化到[0, 1]变成ratio
        reference_points[..., 0:1] = reference_points[..., 0:1] * \
            (pc_range[3] - pc_range[0]) + pc_range[0]  # x
        reference_points[..., 1:2] = reference_points[..., 1:2] * \
            (pc_range[4] - pc_range[1]) + pc_range[1]  # y
        reference_points[..., 2:3] = reference_points[..., 2:3] * \
            (pc_range[5] - pc_range[2]) + pc_range[2]  # z

        # 由(x, y, z)变成(x, y, z, 1)便于与4*4的矩阵相乘
        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)
        # 此时reference_points可以当成是点云的点了.
        reference_points = reference_points.permute(
            1, 0, 2, 3)  # torch.Size([4, 1, 10000, 4])
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)
        # 要往每个相机上去投影. 因此先申请num_cam份.
        # reference_points的shape就变成了, (D, b, num_cam, num_query, 4, 1) 便于和4*4的矩阵做matmul.
        reference_points = reference_points.view(
            int(D), int(B), 1, int(num_query), 4).repeat(1, 1, int(num_cam), 1, 1).unsqueeze(-1)
        # 相机参数由(b,num_cam, 4, 4) 变成(1, b, num_cam, 1, 4, 4) 再变成(D,b,num_cam,num_query,4,4)
        lidar2img = lidar2img.view(
            1, int(B), int(num_cam), 1, 4, 4).repeat(int(D), 1, 1, int(num_query), 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                            reference_points.to(torch.float32)).squeeze(-1)  # torch.Size([4, 1, 6, 10000, 4])
        lidar2img = lidar2img.to(torch.float32)
        reference_points = reference_points.to(torch.float32)
        # torch.Size([4, 1, 6, 10000, 4])
        reference_points_cam = torch.matmul(lidar2img, reference_points)
        reference_points_cam = reference_points_cam.squeeze(-1)
        eps = 1e-5
        # 把每个相机后面的点mask掉. 因为相机后面的点投过来之后第三位是负的.
        mask_zeros = reference_points_cam.new_zeros(
            reference_points_cam.shape[0], reference_points_cam.shape[1], reference_points_cam.shape[2], reference_points_cam.shape[3], 1, dtype=torch.float32)
        mask_ones = mask_zeros + 1
        tpv_mask = torch.where(
            reference_points_cam[..., 2:3] > eps, mask_ones, mask_zeros)
        # 再做齐次化，得到像素坐标
        reference_points_cam = reference_points_cam[..., 0:2] / torch.max(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
        # 由像素坐标转成相对于图像的ratio..
        # NOTE 这里如果不同相机size不一样的话.要除以对应的相机的size
        # select batch_index=0
        reference_points_cam[..., 0] /= img_shape[0][0][1]
        reference_points_cam[..., 1] /= img_shape[0][0][0]
        # 再把超出图像fov范围的点给去掉.
        tpv_mask *= torch.where(
            reference_points_cam[..., 1:2] > 0.0, mask_ones, mask_zeros
        )
        tpv_mask *= torch.where(
            reference_points_cam[..., 1:2] < 1.0, mask_ones, mask_zeros
        )
        tpv_mask *= torch.where(
            reference_points_cam[..., 0:1] < 1.0, mask_ones, mask_zeros
        )
        tpv_mask *= torch.where(
            reference_points_cam[..., 0:1] > 0.0, mask_ones, mask_zeros
        )

        # 再把超出图像fov范围的点给去掉.
        # tpv_mask = (tpv_mask & (reference_points_cam[..., 1:2] > 0.0)
        #             & (reference_points_cam[..., 1:2] < 1.0)
        #             & (reference_points_cam[..., 0:1] < 1.0)
        #             & (reference_points_cam[..., 0:1] > 0.0))  # sum(tpv_mask[0][0][0]) = tensor([1443], device='cuda:0')
        # if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        #     tpv_mask = torch.nan_to_num(tpv_mask)
        # else:
        #     tpv_mask = tpv_mask.new_tensor(
        #         np.nan_to_num(tpv_mask.cpu().numpy()))  # torch.Size([4, 1, 6, 10000, 1])
        tpv_mask = torch.nan_to_num(tpv_mask)

        # 由(D, b, num_cam, num_query, 2) 变成 (num_cam, b, num_query, D, 2)
        reference_points_cam = reference_points_cam.permute(
            2, 1, 3, 0, 4)  # torch.Size([6, 1, 10000, 4, 2])
        # import pdb; pdb.set_trace()
        # tpv_mask = tpv_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
        tpv_mask = tpv_mask.permute(2, 1, 3, 0, 4)
        mask_dim1, mask_dim2, mask_dim3, mask_dim4, mask_dim5 = tpv_mask.shape
        tpv_mask = tpv_mask.reshape(
            mask_dim1, mask_dim2, mask_dim3, mask_dim4*mask_dim5)
        # 至此. reference_points_cam代表的就是像素点相对于各个相机的ratio.
        # tpv_mask就代表哪些点是有效的
        return reference_points_cam, tpv_mask

    def forward_trt(self,
                    # list [tpv_queries_hw, tpv_queries_zh, tpv_queries_wz]
                    tpv_query,
                    key,       # feat_flatten
                    value,     # feat_flatten
                    # *args,
                    tpv_h=None,  # 100
                    tpv_w=None,  # 100
                    tpv_z=None,  # 8
                    tpv_pos=None,  # list [tpv_pos_hw, None, None]
                    spatial_shapes=None,
                    level_start_index=None,
                    lidar2img=None,
                    img_shape=None):
        # ----------------------------------
        # tpv_h = 100
        # tpv_w = 100
        # tpv_z = 8
        # ----------------------------------
        output = tpv_query
        intermediate = []
        # import pdb; pdb.set_trace()
        bs = tpv_query[0].shape[0]
        reference_points_cams, tpv_masks = [], []
        ref_3ds = [self.ref_3d_hw, self.ref_3d_zh, self.ref_3d_wz]
        '''
        self.ref_3d_hw: torch.Size([1, 4, 10000, 3])
        self.ref_3d_zh: torch.Size([1, 32, 800, 3])
        self.ref_3d_wz: torch.Size([1, 32, 800, 3])
        self.pc_range:  [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        '''
        for ref_3d in ref_3ds:
            reference_points_cam, tpv_mask = self.point_sampling_trt(
                ref_3d, self.pc_range, lidar2img, img_shape)  # num_cam, bs, hw++, #p, 2
            reference_points_cams.append(reference_points_cam)
            tpv_masks.append(tpv_mask)
        ref_2d_hw = self.ref_2d_hw.clone().expand(int(bs), -1, -1, -1)
        '''
        self.ref_2d_hw: torch.Size([1, 10000, 1, 2])
        ref_2d_hw: torch.Size([1, 10000, 1, 2])
        '''

        hybird_ref_2d = torch.cat(
            [ref_2d_hw, ref_2d_hw], 0)  # torch.Size([2, 10000, 1, 2])
        for lid, layer in enumerate(self.layers):
            output = layer(
                # len(tpv_query)=3 torch.Size([1, 10000, 256]) torch.Size([1, 800, 256]) torch.Size([1, 800, 256])
                tpv_query,
                key,  # torch.Size([6, 30825, 1, 256])
                value,  # torch.Size([6, 30825, 1, 256])
                # *args,
                tpv_pos=tpv_pos,  # len = 1 torch.Size([1, 10000, 256])
                ref_2d=hybird_ref_2d,  # torch.Size([2, 10000, 1, 2])
                tpv_h=tpv_h,  # 100
                tpv_w=tpv_w,  # 100
                tpv_z=tpv_z,  # 8
                # len=3 torch.Size([6, 1, 10000, 4, 2]) torch.Size([6, 1, 800, 32, 2]) torch.Size([6, 1, 800, 32, 2])
                reference_points_cams=reference_points_cams,
                # len=3  torch.Size([6, 1, 10000, 4]) torch.Size([6, 1, 800, 32])  torch.Size([6, 1, 800, 32])
                tpv_masks=tpv_masks,
                spatial_shapes=spatial_shapes,  # torch.Size([4, 2])
                level_start_index=level_start_index,  # torch.Size([4])
            )
            tpv_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output
