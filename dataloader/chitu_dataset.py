import os
from torch.utils.data import Dataset
import numpy as np
from mmcv.image.io import imread


class ChiTuDataset(Dataset):
    def __init__(self, data_root):
        self.lidar_path = data_root+'lidar'
        self.seg_lidars = data_root + 'lidar_seg'
        self.cam_0_path = data_root+'camera/CAM0'
        self.cam_4_path = data_root+'camera/CAM4'
        self.cam_5_path = data_root+'camera/CAM5'
        self.cam_6_path = data_root+'camera/CAM6'
        self.cam_7_path = data_root+'camera/CAM7'

        self.seg_list = os.listdir(self.seg_lidars)
        self.seg_list.sort()
        self.all_data_info = []
        for label_seg in self.seg_list:
            timestamp = label_seg[:-8]
            points_label_path = os.path.join(self.seg_lidars, label_seg)
            points_label = np.fromfile(
                points_label_path, dtype=np.uint8).reshape([-1, 1])
            CAM0_path = os.path.join(self.cam_0_path, timestamp+".jpg")
            CAM4_path = os.path.join(self.cam_4_path, timestamp+".jpg")
            CAM5_path = os.path.join(self.cam_5_path, timestamp+".jpg")
            CAM6_path = os.path.join(self.cam_6_path, timestamp+".jpg")
            CAM7_path = os.path.join(self.cam_7_path, timestamp+".jpg")
            imgs_paths = [CAM0_path, CAM4_path,
                          CAM5_path, CAM6_path, CAM7_path]
            imgs = []
            for filename in imgs_paths:
                imgs.append(
                    imread(filename, 'unchanged').astype(np.float32)
                )
            points = np.fromfile(os.path.join(
                self.lidar_path, timestamp+".bin"), dtype=np.float32, count=-1).reshape([-1, 3])
            data_meta = self.get_data_meta()
            img_metas = {'lidar2img':data_meta}
            self.all_data_info.append((imgs, img_metas, points, points_label.astype(np.uint8)))
    
    def get_data_meta(self):
        
        cam_0_cam2lidar = [[0.999493, -0.0113972, 0.0297352, 0.130453], [0.0290716, -0.0545346, -
                                                                       0.998089, -0.0990919], [0.012997, 0.998447, -0.0541756, -0.0624945], [0, 0, 0, 1]]
        cam_0_intrinsic = [[909.29440281, 0.0, 653.11572539], [
            0.0, 911.09091142, 349.00353132], [0.0, 0.0, 1.0]]
        cam_0_lidar2camera_r = np.linalg.inv(
            np.array(cam_0_cam2lidar)[:3, :3])
        cam_0_lidar2camera_t = (
            np.array(cam_0_cam2lidar)[:3, -1] @ cam_0_lidar2camera_r.T
        )
        cam_0_lidar2camera_rt = np.eye(4).astype(np.float32)
        cam_0_lidar2camera_rt[:3, :3] = cam_0_lidar2camera_r.T
        cam_0_lidar2camera_rt[3, :3] = -cam_0_lidar2camera_t
        cam_0_viewpad = np.eye(4)
        cam_0_viewpad[:np.array(cam_0_intrinsic).shape[0], :np.array(cam_0_intrinsic).shape[1]] = cam_0_intrinsic
        cam_0_lidar2img_rt = (cam_0_viewpad @ cam_0_lidar2camera_rt.T)
        # -------------------------------------------------------------------------------------------------------
        cam_4_cam2lidar = [[0.999493, -0.0113972, 0.0297352, 0.130453], [0.0290716, -0.0545346, -
                                                                       0.998089, -0.0990919], [0.012997, 0.998447, -0.0541756, -0.0624945], [0, 0, 0, 1]]
        cam_4_intrinsic = [[909.29440281, 0.0, 653.11572539], [
            0.0, 911.09091142, 349.00353132], [0.0, 0.0, 1.0]]
        cam_4_lidar2camera_r = np.linalg.inv(
            np.array(cam_4_cam2lidar)[:3, :3])
        cam_4_lidar2camera_t = (
            np.array(cam_0_cam2lidar)[:3, -1] @ cam_4_lidar2camera_r.T
        )
        cam_4_lidar2camera_rt = np.eye(4).astype(np.float32)
        cam_4_lidar2camera_rt[:3, :3] = cam_4_lidar2camera_r.T
        cam_4_lidar2camera_rt[3, :3] = -cam_4_lidar2camera_t
        cam_4_viewpad = np.eye(4)
        cam_4_viewpad[:np.array(cam_4_intrinsic).shape[0], :np.array(cam_4_intrinsic).shape[1]] = cam_4_intrinsic
        cam_4_lidar2img_rt = (cam_4_viewpad @ cam_4_lidar2camera_rt.T)
        # ------------------------------------------------------------------------------------------------------
        cam_5_cam2lidar = [[0.999493, -0.0113972, 0.0297352, 0.130453], [0.0290716, -0.0545346, -
                                                                       0.998089, -0.0990919], [0.012997, 0.998447, -0.0541756, -0.0624945], [0, 0, 0, 1]]
        cam_5_intrinsic = [[909.29440281, 0.0, 653.11572539], [
            0.0, 911.09091142, 349.00353132], [0.0, 0.0, 1.0]]
        cam_5_lidar2camera_r = np.linalg.inv(
            np.array(cam_5_cam2lidar)[:3, :3])
        cam_5_lidar2camera_t = (
            np.array(cam_5_cam2lidar)[:3, -1] @ cam_5_lidar2camera_r.T
        )
        cam_5_lidar2camera_rt = np.eye(4).astype(np.float32)
        cam_5_lidar2camera_rt[:3, :3] = cam_5_lidar2camera_r.T
        cam_5_lidar2camera_rt[3, :3] = -cam_5_lidar2camera_t
        cam_5_viewpad = np.eye(4)
        cam_5_viewpad[:np.array(cam_5_intrinsic).shape[0], :np.array(cam_5_intrinsic).shape[1]] = cam_5_intrinsic
        cam_5_lidar2img_rt = (cam_5_viewpad @ cam_5_lidar2camera_rt.T)
        # -------------------------------------------------------------------------------------------------------
        cam_6_cam2lidar = [[0.999493, -0.0113972, 0.0297352, 0.130453], [0.0290716, -0.0545346, -
                                                                       0.998089, -0.0990919], [0.012997, 0.998447, -0.0541756, -0.0624945], [0, 0, 0, 1]]
        cam_6_intrinsic = [[909.29440281, 0.0, 653.11572539], [
            0.0, 911.09091142, 349.00353132], [0.0, 0.0, 1.0]]
        cam_6_lidar2camera_r = np.linalg.inv(
            np.array(cam_6_cam2lidar)[:3, :3])
        cam_6_lidar2camera_t = (
            np.array(cam_6_cam2lidar)[:3, -1] @ cam_6_lidar2camera_r.T
        )
        cam_6_lidar2camera_rt = np.eye(4).astype(np.float32)
        cam_6_lidar2camera_rt[:3, :3] = cam_6_lidar2camera_r.T
        cam_6_lidar2camera_rt[3, :3] = -cam_6_lidar2camera_t
        cam_6_viewpad = np.eye(4)
        cam_6_viewpad[:np.array(cam_6_intrinsic).shape[0], :np.array(cam_6_intrinsic).shape[1]] = cam_6_intrinsic
        cam_6_lidar2img_rt = (cam_6_viewpad @ cam_6_lidar2camera_rt.T)
        # --------------------------------------------------------------------------------------------------------
        cam_7_cam2lidar = [[0.999493, -0.0113972, 0.0297352, 0.130453], [0.0290716, -0.0545346, -
                                                                       0.998089, -0.0990919], [0.012997, 0.998447, -0.0541756, -0.0624945], [0, 0, 0, 1]]
        cam_7_intrinsic = [[909.29440281, 0.0, 653.11572539], [
            0.0, 911.09091142, 349.00353132], [0.0, 0.0, 1.0]]
        cam_7_lidar2camera_r = np.linalg.inv(
            np.array(cam_0_cam2lidar)[:3, :3])
        cam_7_lidar2camera_t = (
            np.array(cam_7_cam2lidar)[:3, -1] @ cam_7_lidar2camera_r.T
        )
        cam_7_lidar2camera_rt = np.eye(4).astype(np.float32)
        cam_7_lidar2camera_rt[:3, :3] = cam_7_lidar2camera_r.T
        cam_7_lidar2camera_rt[3, :3] = -cam_7_lidar2camera_t
        cam_7_viewpad = np.eye(4)
        cam_7_viewpad[:np.array(cam_7_intrinsic).shape[0], :np.array(cam_7_intrinsic).shape[1]] = cam_7_intrinsic
        cam_7_lidar2img_rt = (cam_7_viewpad @ cam_7_lidar2camera_rt.T)

        lidar2cam_rts = [cam_0_lidar2img_rt, cam_4_lidar2img_rt, cam_5_lidar2img_rt, cam_6_lidar2img_rt, cam_7_lidar2img_rt]
        return lidar2cam_rts
    
    def __getitem__(self, index):
        return self.all_data_info[index]
    
    def __len__(self):
        return len(self.all_data_info)


'''
["car", "pedestrian", "person", "bus", "truck", "bicycle", "motorcycle", "rider",
"bicycle_motorcycle", "forklift", "trailer", "rack", "shelves", "traffic_cone",
"goods", "cargo", "traffic_light", "other_vehicle",]

1:(person, pedestrian), 2:(bicycle, motorcycle, rider) 3:(car)  4:(bus) 5:(truck) 6:(forklift) 7:(trailer)
8:(rack) 9:(shelves)  10:(traffic_cone) 11:(goods, cargo) 12:(traffic_light) 13:(other_vehicle)
'''
