from .tpvformer import TPVFormer
from mmseg.models import SEGMENTORS, builder
from mmcv.runner import force_fp32, auto_fp16
import torch

@SEGMENTORS.register_module()
class TPVFormerTRT(TPVFormer):
    def __init__(self, *args, **kwargs):
        super(TPVFormerTRT, self).__init__(*args, **kwargs)

    @auto_fp16(apply_to=('img'))
    def extract_img_feat(self, img, use_grid_mask=None):
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)

        if use_grid_mask is None:
            use_grid_mask = self.use_grid_mask  # True
        if use_grid_mask:
            img = self.grid_mask(img)
        # -----------------------------------------------------------------------------------
        # torch.onnx.export(self.img_backbone, img,
        #                 "/home/zs/Code/TPVFormer_new/tpvformer_backbone.onnx",
        #                 verbose=False,
        #                 opset_version=17,
        #                 input_names=["imgs"],
        #                 output_names=["output1", "output2", "output3"], autograd_inlining=False)
                        
        # print("TPVFormer_backbone Convert to ONNX Done!")
        # --------------------------------------------------------------------------------------
        img_feats = self.img_backbone(img)
        # import pdb; pdb.set_trace()
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        # import pdb; pdb.set_trace()
        if hasattr(self, 'img_neck'):
            # ------------------------------------------------------------------------------------
            # neck_input1 = img_feats[0].cpu().numpy()
            # neck_input2 = img_feats[1].cpu().numpy()
            # neck_input3 = img_feats[2].cpu().numpy()
            # import numpy as np
            # np.save("./neck_input1.npy", neck_input1)
            # np.save("./neck_input2.npy", neck_input2)
            # np.save("./neck_input3.npy", neck_input3)
            # torch.onnx.export(self.img_neck, [img_feats[0], img_feats[1], img_feats[2]], # torch.Size([6, 512, 116, 200]), torch.Size([6, 1024, 58, 100]), torch.Size([6, 2048, 29, 50])
            #             "/home/zs/Code/TPVFormer_new/tpvformer_neck.onnx",
            #             verbose=False,
            #             opset_version=17,
            #             input_names=["input1", "input2", "input3"],
            #             output_names=["output1", "output2", "output3", "output4"], 
            #             dynamic_axes={"input1": {0: "num_cam", 2: "h1", 3:"w1"}, "input2": {0:"num_cam", 2: "h2", 3:"w2"}, "input3": {0: "num_cam", 2:"h3", 3:"w3"}},
            #             autograd_inlining=False)
                        
            # print("TPVFormer_neck Convert to ONNX Done!")
            # import pdb; pdb.set_trace()
            # ------------------------------------------------------------------------------------
            img_feats = self.img_neck(img_feats)  # fpn    torch.Size([6, 256, 116, 200]), torch.Size([6, 256, 58, 100]), torch.Size([6, 256, 29, 50]), torch.Size([6, 256, 15, 25])
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def forward_trt(self, img, points, lidar2img, img_shape):
    # def forward_trt(self, img, lidar2img, img_shape):
        img_feats = self.extract_img_feat(img=img)
        '''
        extract_img_feat使用backbone和neck(fpn)对图像进行特征提取，最后得到四个特征图大小分别为:
        img_feats[0]: torch.Size([1, 6, 256, 116, 200])
        img_feats[1]: torch.Size([1, 6, 256, 58, 100])
        img_feats[2]: torch.Size([1, 6, 256, 29, 50])
        img_feats[3]: torch.Size([1, 6, 256, 15, 25])
        '''
        outs = self.tpv_head.forward_trt(img_feats, lidar2img, img_shape)  # outs[0]: torch.Size([1, 10000, 256])  outs[1]: torch.Size([1, 800, 256])  outs[2]:torch.Size([1, 800, 256])
        outs = self.tpv_aggregator.forward_trt(outs, points) # outs[0]: torch.Size([1, 18, 100, 100, 8])  outs[1]: torch.Size([1, 18, 34752, 1, 1])  
        return outs
    
    def forward_trt_no_points(self, img, lidar2img, img_shape):
        img_feats = self.extract_img_feat(img=img)
        outs = self.tpv_head.forward_trt(img_feats, lidar2img, img_shape)  # outs[0]: torch.Size([1, 10000, 256])  outs[1]: torch.Size([1, 800, 256])  outs[2]:torch.Size([1, 800, 256])
        outs = self.tpv_aggregator.forward_trt_no_points(outs) # outs[0]: torch.Size([1, 18, 100, 100, 8])  outs[1]: torch.Size([1, 18, 34752, 1, 1])  
        return outs
