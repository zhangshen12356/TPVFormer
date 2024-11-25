
from mayavi import mlab
mlab.options.offscreen = False
print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))

import argparse, torch, os, json
import shutil
import numpy as np
import mmcv
from mmcv import Config
from collections import OrderedDict

# from pyvirtualdisplay import Display
# display = Display(visible=False, size=(2560, 1440))
# display.start()


def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v for k, v in state_dict.items()})
    return state_dict


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) # [0, 1, ..., 256]
    # g_xx = g_xx[::-1]
    g_yy = np.arange(0, dims[1]) # [0, 1, ..., 256]
    # g_yy = g_yy[::-1]
    g_zz = np.arange(0, dims[2]) # [0, 1, ..., 32]

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid


def draw(
    voxels,          # semantic occupancy predictions  w, h, z
    pred_pts,        # lidarseg predictions   (34752,)
    vox_origin,
    voxel_size=0.2,  # voxel size in the real world
    grid=None,       # voxel coordinates of point cloud
    pt_label=None,   # label of point cloud
    save_dirs=None,
    cam_positions=None,
    focal_positions=None,
    timestamp=None,
    mode=0,
):
    w, h, z = voxels.shape  # (100, 100, 8)
    grid = grid.astype(np.int)  # 表示的是原始点云 (34752, 3)

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size # [1.024, 1.024, 1.0]
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    if mode == 0:
        grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    
        # draw a simple car at the middle
        # car_vox_range = np.array([
        #     [w//2 - 2 - 4, w//2 - 2 + 4],
        #     [h//2 - 2 - 4, h//2 - 2 + 4],
        #     [z//2 - 2 - 3, z//2 - 2 + 3]
        # ], dtype=np.int)
        # car_x = np.arange(car_vox_range[0, 0], car_vox_range[0, 1])
        # car_y = np.arange(car_vox_range[1, 0], car_vox_range[1, 1])
        # car_z = np.arange(car_vox_range[2, 0], car_vox_range[2, 1])
        # car_xx, car_yy, car_zz = np.meshgrid(car_x, car_y, car_z)
        # car_label = np.zeros([8, 8, 6], dtype=np.int)
        # car_label[:3, :, :2] = 17
        # car_label[3:6, :, :2] = 18
        # car_label[6:, :, :2] = 19
        # car_label[:3, :, 2:4] = 18
        # car_label[3:6, :, 2:4] = 19
        # car_label[6:, :, 2:4] = 17
        # car_label[:3, :, 4:] = 19
        # car_label[3:6, :, 4:] = 17
        # car_label[6:, :, 4:] = 18
        # car_grid = np.array([car_xx.flatten(), car_yy.flatten(), car_zz.flatten()]).T
        # car_indexes = car_grid[:, 0] * h * z + car_grid[:, 1] * z + car_grid[:, 2]
        # grid_coords[car_indexes, 3] = car_label.flatten()
    
    elif mode == 1:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        pred_pts = pred_pts[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, pred_pts.reshape(-1)]).T
    elif mode == 2:
        indexes = grid[:, 0] * h * z + grid[:, 1] * z + grid[:, 2]
        indexes, pt_index = np.unique(indexes, return_index=True)
        gt_label = pt_label[pt_index]
        grid_coords = grid_coords[indexes]
        grid_coords = np.vstack([grid_coords.T, gt_label.reshape(-1)]).T
    else:
        raise NotImplementedError
    # grid_coords[grid_coords[:, 3] == 15, 3] = 20

    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    # fov_voxels = fov_grid_coords[(fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)]
    fov_voxels = fov_grid_coords[(fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 14)]
    # import pdb; pdb.set_trace()
    print(len(fov_voxels))
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 1],
        fov_voxels[:, 0],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=0.95 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19, # 16
    )
    '''
    1:(person, pedestrian), 2:(bicycle, motorcycle, rider) 3:(car)  4:(bus) 5:(truck) 6:(forklift) 7:(trailer)
    8:(rack) 9:(shelves)  10:(traffic_cone) 11:(goods, cargo) 12:(traffic_light) 13:(other_vehicle)
'''

    colors = np.array(
        [
            [255, 120,  50, 255],       # person ,pedestrian              orange
            [255, 192, 203, 255],       # bicycle ,motorcycle, rider      pink
            [255, 255,   0, 255],       # car                             yellow
            [  0, 150, 245, 255],       # bus                             blue
            [  0, 255, 255, 255],       # truck                           cyan
            [255, 127,   0, 255],       # forklift                        dark orange
            [255,   0,   0, 255],       # trailer                         red
            [255, 240, 150, 255],       # rack                            light yellow
            [135,  60,   0, 255],       # shelves                         brown
            [160,  32, 240, 255],       # traffic_cone                    purple                
            [255,   0, 255, 255],       # goods, cargo                    dark pink
            # [175,   0,  75, 255],     # other_flat                      dark red
            [139, 137, 137, 255],
            [ 75,   0,  75, 255],       # traffic_light                   dard purple
            [150, 240,  80, 255],       # other_vehicle                   light green          
            [230, 230, 250, 255],       #                                 white
            [  0, 175,   0, 255],       #                                 green
            [  0, 255, 127, 255],       # ego car                         dark cyan
            [255,  99,  71, 255],       # ego car
            [  0, 191, 255, 255]        # ego car
        ]
    ).astype(np.uint8)
    
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    scene = figure.scene
    
    scene.camera.position = [  0.75131739, -35.08337438,  16.71378558]
    scene.camera.focal_point = [  0.75131739, -34.21734897,  16.21378558]
    scene.camera.view_angle = 40.0
    scene.camera.view_up = [0.0, 0.0, 1.0]
    scene.camera.clipping_range = [0.01, 300.]
    scene.camera.compute_view_plane_normal()
    scene.render()

    # scene.camera.position = [ 0.75131739,  0.78265103, 93.21378558]
    # scene.camera.focal_point = [ 0.75131739,  0.78265103, 92.21378558]
    # scene.camera.view_angle = 40.0
    # scene.camera.view_up = [0., 1., 0.]
    # scene.camera.clipping_range = [0.01, 400.]
    # scene.camera.compute_view_plane_normal()
    # scene.render()
    
    mlab.savefig('chitu_points.jpg')  # !!!
    # mlab.show()


if __name__ == "__main__":
    import sys; sys.path.insert(0, os.path.abspath('.'))

    device = torch.device('cuda:0')
    ## prepare config
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_occupancy_chitu.py')
    # parser.add_argument('--py-config', default='config/tpv04_occupancy.py')
    parser.add_argument('--work-dir', type=str, default='out/tpv04_occupancy_chitu')
    parser.add_argument('--ckpt-path', type=str, default='/home/zs/Code/TPVFormer_new/out/tpv04_occupancy_chitu/epoch_24.pth')
    # parser.add_argument('--ckpt-path', type=str, default='/home/zs/Code/TPVFormer_new/ckpts/tpv04_occupancy_v2.pth')
    parser.add_argument('--vis-train', action='store_true', default=False)
    parser.add_argument('--save-path', type=str, default='out/tpv_occupancy/frames')
    parser.add_argument('--frame-idx', type=int, default=[4], nargs='+', 
                        help='idx of frame to visualize, the idx corresponds to the order in pkl file.')
    parser.add_argument('--mode', type=int, default=0, help='0: occupancy, 1: predicted point cloud, 2: gt point cloud')

    args = parser.parse_args()
    print(args)

    cfg = Config.fromfile(args.py_config)
    dataset_config = cfg.dataset_params

    # prepare model
    logger = mmcv.utils.get_logger('mmcv')
    logger.setLevel("WARNING")
    if cfg.get('occupancy', False):
        from builder import tpv_occupancy_builder as model_builder
    else:
        from builder import tpv_lidarseg_builder as model_builder
    my_model = model_builder.build(cfg.model).to(device)
    if args.ckpt_path:
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        print(my_model.load_state_dict(revise_ckpt(ckpt)))
    my_model.eval()

    # prepare data
    from visualization.dataset_chitu import ChiTuDataset_vis, DatasetWrapper_Chitu_vis
    
    data_path = 'data/chitu/'

    pt_dataset = ChiTuDataset_vis(data_path)

    dataset = DatasetWrapper_Chitu_vis(
        pt_dataset,
        grid_size=cfg.grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        phase='val'
    )
    print(len(dataset))

    for index in args.frame_idx:
        print(f'processing frame {index}')
        batch_data, filelist = dataset[index]
        print(filelist)
        imgs, img_metas, vox_label, grid, pt_label = batch_data
        # -------------------------------------------
        # im0, im4, im5, im6, im7 = imgs
        # imgs[0] = im0
        # imgs[1] = im7
        # imgs[2] = im5
        # imgs[3] = im4
        # imgs[4] = im6
        # lida2im0, lida2im4, lida2im5, lida2im6, lida2im7 = img_metas['lidar2img']
        # img_metas['lidar2img'][0] = lida2im0
        # img_metas['lidar2img'][1] = lida2im7
        # img_metas['lidar2img'][2] = lida2im5
        # img_metas['lidar2img'][3] = lida2im4
        # img_metas['lidar2img'][4] = lida2im6
        
        # dumy_img = np.zeros((3, 736, 1280))
        # dumy_lidar2img = np.zeros((4, 4))
        # # imgs.append(dumy_img)
        # imgs.append(imgs[-1])
        # # img_metas['lidar2img'].append(dumy_lidar2img)
        # img_metas['lidar2img'].append(img_metas['lidar2img'][-1])
        # --------------------------------------------
        # import pdb; pdb.set_trace()
        imgs = torch.from_numpy(np.stack([imgs]).astype(np.float32)).to(device)
        grid = torch.from_numpy(np.stack([grid]).astype(np.float32)).to(device)
        with torch.no_grad():
            # outputs_vox, outputs_pts = my_model(img=imgs, img_metas=[img_metas], points=grid.clone())
            outputs_vox= my_model(img=imgs, img_metas=[img_metas], points=None)
            '''
            outputs_vox: torch.Size([1, 18, 100, 100, 8])
            outputs_pts: torch.Size([1, 18, 34752, 1, 1])
            '''
            predict_vox = torch.argmax(outputs_vox, dim=1) # bs, w, h, z
            predict_vox = predict_vox.squeeze(0).cpu().numpy() # w, h, z

            # predict_pts = torch.argmax(outputs_pts, dim=1) # bs, n, 1, 1
            # predict_pts = predict_pts.squeeze().cpu().numpy() # n  (34752,)
            
        # predict_pts[np.where(predict_pts==14)] = 0
        # predict_pts = predict_pts.astype(np.uint8) 
        # predict_pts.tofile(filelist[0].split("/")[-1][:-4]+"_predict.bin")
        voxel_origin = dataset_config['min_volume_space']  # [-51.2, -51.2, -5]
        voxel_max = dataset_config['max_volume_space']     # [51.2, 51.2, 3]
        grid_size = cfg.grid_size                          # [100, 100, 8]
        resolution = [(e - s) / l for e, s, l in zip(voxel_max, voxel_origin, grid_size)]
        # [1.024, 1.024, 1.0]

        frame_dir = os.path.join(args.save_path, str(index))
        os.makedirs(frame_dir, exist_ok=True)
        
        for idx, filename in enumerate(filelist):          
            shutil.copy(filename, os.path.join(frame_dir, str(idx)+"_"+os.path.basename(filename)))

        draw(predict_vox, 
            #  predict_pts,
             None,
             voxel_origin, 
             resolution, 
             grid.squeeze(0).cpu().numpy(), 
             pt_label.squeeze(-1),
             frame_dir,
             img_metas['cam_positions'],
             img_metas['focal_positions'],
             timestamp='0000000',
             mode=args.mode)
