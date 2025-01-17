
import torch, torch.nn as nn, torch.nn.functional as F
from mmcv.runner import BaseModule
from mmseg.models import HEADS
from mmcv.ops.point_sample import bilinear_grid_sample


@HEADS.register_module()
class TPVAggregator(BaseModule):
    def __init__(
        self, tpv_h, tpv_w, tpv_z, nbr_classes=20, 
        in_dims=64, hidden_dims=128, out_dims=None,
        scale_h=2, scale_w=2, scale_z=2, use_checkpoint=False
    ):
        super().__init__()
        self.tpv_h = tpv_h
        self.tpv_w = tpv_w
        self.tpv_z = tpv_z
        self.scale_h = scale_h
        self.scale_w = scale_w
        self.scale_z = scale_z

        out_dims = in_dims if out_dims is None else out_dims

        self.decoder = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.Softplus(),
            nn.Linear(hidden_dims, out_dims)
        )

        self.classifier = nn.Linear(out_dims, nbr_classes)
        self.classes = nbr_classes
        self.use_checkpoint = use_checkpoint
    
    def forward(self, tpv_list, points=None):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)  # torch.Size([1, 256, 100, 100])
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)

        if self.scale_h != 1 or self.scale_w != 1:
            tpv_hw = F.interpolate(
                tpv_hw, 
                size=(self.tpv_h*self.scale_h, self.tpv_w*self.scale_w),
                mode='bilinear'
            )
        if self.scale_z != 1 or self.scale_h != 1:
            tpv_zh = F.interpolate(
                tpv_zh, 
                size=(self.tpv_z*self.scale_z, self.tpv_h*self.scale_h),
                mode='bilinear'
            )
        if self.scale_w != 1 or self.scale_z != 1:
            tpv_wz = F.interpolate(
                tpv_wz, 
                size=(self.tpv_w*self.scale_w, self.tpv_z*self.scale_z),
                mode='bilinear'
            )
        
        if points is not None and bs==1:
            # points: bs, n, 3
            _, n, _ = points.shape
            points = points.reshape(bs, 1, n, 3)
            # 将点云投影在3个bev的特征上
            points[..., 0] = points[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
            points[..., 1] = points[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
            points[..., 2] = points[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1
            # 根据点云的坐标在特征上进行采样
            sample_loc = points[:, :, :, [0, 1]]  # torch.Size([1, 1, 34752, 2])
            tpv_hw_pts = F.grid_sample(tpv_hw, sample_loc).squeeze(2) # bs, c, n
            sample_loc = points[:, :, :, [1, 2]]
            tpv_zh_pts = F.grid_sample(tpv_zh, sample_loc).squeeze(2)
            sample_loc = points[:, :, :, [2, 0]]
            tpv_wz_pts = F.grid_sample(tpv_wz, sample_loc).squeeze(2)

            tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z) # torch.Size([1, 256, 100, 100, 8])
            tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1) # torch.Size([1, 256, 100, 100, 8])
            tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1) # torch.Size([1, 256, 100, 100, 8])
        
            fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2)     # torch.Size([1, 256, 80000])
            fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts                  # torch.Size([1, 256, 34752])
            fused = torch.cat([fused_vox, fused_pts], dim=-1) # bs, c, whz+n  # torch.Size([1, 256, 114752])
            
            fused = fused.permute(0, 2, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)      # torch.Size([1, 114752, 256])
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)  # torch.Size([1, 114752, 18])
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 2, 1)  # torch.Size([1, 18, 114752])
            logits_vox = logits[:, :, :(-n)].reshape(bs, self.classes, self.scale_w*self.tpv_w, self.scale_h*self.tpv_h, self.scale_z*self.tpv_z)  # torch.Size([1, 18, 100, 100, 8])
            logits_pts = logits[:, :, (-n):].reshape(bs, self.classes, n, 1, 1)  # torch.Size([1, 18, 34752, 1, 1])
            return logits_vox, logits_pts
            
        else:
            tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
            tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1)
            tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1)
        
            fused = tpv_hw + tpv_zh + tpv_wz
            fused = fused.permute(0, 2, 3, 4, 1)
            if self.use_checkpoint:
                fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
                logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
            else:
                fused = self.decoder(fused)
                logits = self.classifier(fused)
            logits = logits.permute(0, 4, 1, 2, 3)
        
            return logits

    def forward_trt(self, tpv_list, points=None):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)
        # import pdb; pdb.set_trace()
        # if self.scale_h != 1 or self.scale_w != 1:
        #     tpv_hw = F.interpolate(
        #         tpv_hw, 
        #         size=(self.tpv_h*self.scale_h, self.tpv_w*self.scale_w),
        #         mode='bilinear'
        #     )
        # if self.scale_z != 1 or self.scale_h != 1:
        #     tpv_zh = F.interpolate(
        #         tpv_zh, 
        #         size=(self.tpv_z*self.scale_z, self.tpv_h*self.scale_h),
        #         mode='bilinear'
        #     )
        # if self.scale_w != 1 or self.scale_z != 1:
        #     tpv_wz = F.interpolate(
        #         tpv_wz, 
        #         size=(self.tpv_w*self.scale_w, self.tpv_z*self.scale_z),
        #         mode='bilinear'
        #     )
        
        
        # points: bs, n, 3
        _, n, _ = points.shape
        points = points.reshape(bs, 1, n, 3)
        points[..., 0] = points[..., 0] / (self.tpv_w*self.scale_w) * 2 - 1
        points[..., 1] = points[..., 1] / (self.tpv_h*self.scale_h) * 2 - 1
        points[..., 2] = points[..., 2] / (self.tpv_z*self.scale_z) * 2 - 1
        sample_loc = points[:, :, :, [0, 1]]
     
        tpv_hw_pts = bilinear_grid_sample(tpv_hw, sample_loc).view(bs, c, -1) # bs, c, n  torch.Size([1, 256, 34752])
        sample_loc = points[:, :, :, [1, 2]]
        tpv_zh_pts = bilinear_grid_sample(tpv_zh, sample_loc).view(bs, c, -1)  # torch.Size([1, 256, 34752])
        sample_loc = points[:, :, :, [2, 0]]
        tpv_wz_pts = bilinear_grid_sample(tpv_wz, sample_loc).view(bs, c, -1)   # torch.Size([1, 256, 34752])
      

        tpv_hw_vox = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
        tpv_zh_vox = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1)
        tpv_wz_vox = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1)
    
        fused_vox = (tpv_hw_vox + tpv_zh_vox + tpv_wz_vox).flatten(2)
        fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts
        fused = torch.cat([fused_vox, fused_pts], dim=-1) # bs, c, whz+n
        
        fused = fused.permute(0, 2, 1)
        # if self.use_checkpoint:
        #     fused = torch.utils.checkpoint.checkpoint(self.decoder, fused)
        #     logits = torch.utils.checkpoint.checkpoint(self.classifier, fused)
        # else:
        fused = self.decoder(fused)
        logits = self.classifier(fused)
        logits = logits.permute(0, 2, 1)
        logits_vox = logits[:, :, :(-n)].reshape(bs, self.classes, self.scale_w*self.tpv_w, self.scale_h*self.tpv_h, self.scale_z*self.tpv_z)
        logits_pts = logits[:, :, (-n):].reshape(bs, self.classes, n, 1, 1)
        return logits_vox, logits_pts
    
    def forward_trt_no_points(self, tpv_list):
        """
        tpv_list[0]: bs, h*w, c
        tpv_list[1]: bs, z*h, c
        tpv_list[2]: bs, w*z, c
        """
        tpv_hw, tpv_zh, tpv_wz = tpv_list[0], tpv_list[1], tpv_list[2]
        bs, _, c = tpv_hw.shape
        tpv_hw = tpv_hw.permute(0, 2, 1).reshape(bs, c, self.tpv_h, self.tpv_w)
        tpv_zh = tpv_zh.permute(0, 2, 1).reshape(bs, c, self.tpv_z, self.tpv_h)
        tpv_wz = tpv_wz.permute(0, 2, 1).reshape(bs, c, self.tpv_w, self.tpv_z)
        # if self.scale_h != 1 or self.scale_w != 1:
        #     tpv_hw = F.interpolate(
        #         tpv_hw, 
        #         size=(self.tpv_h*self.scale_h, self.tpv_w*self.scale_w),
        #         mode='bilinear'
        #     )
        # if self.scale_z != 1 or self.scale_h != 1:
        #     tpv_zh = F.interpolate(
        #         tpv_zh, 
        #         size=(self.tpv_z*self.scale_z, self.tpv_h*self.scale_h),
        #         mode='bilinear'
        #     )
        # if self.scale_w != 1 or self.scale_z != 1:
        #     tpv_wz = F.interpolate(
        #         tpv_wz, 
        #         size=(self.tpv_w*self.scale_w, self.tpv_z*self.scale_z),
        #         mode='bilinear'
        #     )
        
        
        # points: bs, n, 3
        tpv_hw = tpv_hw.unsqueeze(-1).permute(0, 1, 3, 2, 4).expand(-1, -1, -1, -1, self.scale_z*self.tpv_z)
        tpv_zh = tpv_zh.unsqueeze(-1).permute(0, 1, 4, 3, 2).expand(-1, -1, self.scale_w*self.tpv_w, -1, -1)
        tpv_wz = tpv_wz.unsqueeze(-1).permute(0, 1, 2, 4, 3).expand(-1, -1, -1, self.scale_h*self.tpv_h, -1)
    
        fused = tpv_hw + tpv_zh + tpv_wz
        fused = fused.permute(0, 2, 3, 4, 1)
        fused = self.decoder(fused)
        logits = self.classifier(fused)
        logits = logits.permute(0, 4, 1, 2, 3)
    
        return logits 
