
import os
import numpy as np
from torch.utils import data
import yaml
import pickle
from mmcv.image.io import imread
from copy import deepcopy

import torch
import numba as nb
from torch.utils import data
from dataloader.transform_3d import PadMultiViewImage, \
    NormalizeMultiviewImage, \
    PhotoMetricDistortionMultiViewImage


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

train_pipeline = [
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]

test_pipeline = [
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
]


class ChiTuDataset_vis(data.Dataset):
    def __init__(self, data_root):
        self.lidar_path = data_root+'lidar'
        self.seg_lidars = data_root + 'lidar_seg'
        self.cam_0_path = data_root+'camera/CAM0_undistort'
        self.cam_4_path = data_root+'camera/CAM4_undistort'
        self.cam_5_path = data_root+'camera/CAM5_undistort'
        self.cam_6_path = data_root+'camera/CAM6_undistort'
        self.cam_7_path = data_root+'camera/CAM7_undistort'

        self.seg_list = os.listdir(self.seg_lidars)
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
            lidar2img, cam_positions, focal_positions = self.get_data_meta()
            img_metas = {'lidar2img':lidar2img, 'cam_positions':cam_positions, 'focal_positions':focal_positions}
            self.all_data_info.append((imgs, img_metas, points, points_label.astype(np.uint8), imgs_paths))
    
    def get_data_meta(self):
        f = 0.0055
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
        cam_0_cam_position = np.linalg.inv(cam_0_lidar2camera_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
        cam_0_focal_position = np.linalg.inv(cam_0_lidar2camera_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
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
        cam_4_cam_position = np.linalg.inv(cam_4_lidar2camera_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
        cam_4_focal_position = np.linalg.inv(cam_4_lidar2camera_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
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
        cam_5_cam_position = np.linalg.inv(cam_5_lidar2camera_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
        cam_5_focal_position = np.linalg.inv(cam_5_lidar2camera_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
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
        cam_6_cam_position = np.linalg.inv(cam_6_lidar2camera_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
        cam_6_focal_position = np.linalg.inv(cam_6_lidar2camera_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
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
        cam_7_cam_position = np.linalg.inv(cam_7_lidar2camera_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
        cam_7_focal_position = np.linalg.inv(cam_7_lidar2camera_rt.T) @ np.array([0., 0., f, 1.]).reshape([4, 1])
        
        lidar2cam_rts = [cam_0_lidar2img_rt, cam_4_lidar2img_rt, cam_5_lidar2img_rt, cam_6_lidar2img_rt, cam_7_lidar2img_rt]
        cam_positions = [cam_0_cam_position.flatten()[:3], cam_4_cam_position.flatten()[:3], cam_5_cam_position.flatten()[:3], cam_6_cam_position.flatten()[:3], cam_7_cam_position.flatten()[:3]]
        focal_positions = [cam_0_focal_position, cam_4_focal_position, cam_5_focal_position, cam_6_focal_position, cam_7_focal_position]
        return lidar2cam_rts, cam_positions, focal_positions
    
    def __getitem__(self, index):
        return self.all_data_info[index]
    
    def __len__(self):
        return len(self.all_data_info)
    

class DatasetWrapper_Chitu_vis(data.Dataset):
    def __init__(self, in_dataset, grid_size, ignore_label=0, fixed_volume_space=False, 
                 max_volume_space=[50, np.pi, 3], min_volume_space=[0, -np.pi, -5], phase='train'):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size).astype(np.int32)
        self.ignore_label = ignore_label
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.polar = False

        if phase == 'train':
            transforms = [
                PhotoMetricDistortionMultiViewImage(),
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        else:
            transforms = [
                NormalizeMultiviewImage(**img_norm_cfg),
                PadMultiViewImage(size_divisor=32)
            ]
        self.transforms = transforms

    def __len__(self):
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        data = self.point_cloud_dataset[index]
        imgs, img_metas, xyz, labels, filelist = data
        # deal with img augmentations
        imgs_dict = {'img': imgs}
        for t in self.transforms:
            imgs_dict = t(imgs_dict)
        imgs = imgs_dict['img']
        imgs = [img.transpose(2, 0, 1) for img in imgs]
        img_metas['img_shape'] = imgs_dict['img_shape']

        xyz_pol = xyz
        
        assert self.fixed_volume_space
        max_bound = np.asarray(self.max_volume_space)  # 51.2 51.2 3
        min_bound = np.asarray(self.min_volume_space)  # -51.2 -51.2 -5
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size                 # 200, 200, 16
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        # TODO: grid_ind of float dtype may be better.
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (imgs, img_metas, processed_label, grid_ind, labels)

        return data_tuple, filelist


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label
