import os
import onnx
import numpy as np
import onnxruntime as ort
import mmcv
from mmcv import Config
from collections import OrderedDict
import torch
import argparse
# from mmcv.ops import get_onnxruntime_op_path
# from mmdeploy.ops import get_onnxruntime_op_path

def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v for k, v in state_dict.items()})
    return state_dict

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath('.'))

    device = torch.device('cuda:0')
    # device = torch.device('cpu')
    # prepare config
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_occupancy_trt.py')
    parser.add_argument('--work-dir', type=str, default='out/tpv_occupancy')
    parser.add_argument('--ckpt-path', type=str,
                        default='/home/zs/Code/TPVFormer_new/ckpts/tpv04_occupancy_v2.pth')
    parser.add_argument('--vis-train', action='store_true', default=False)
    parser.add_argument('--save-path', type=str,
                        default='out/tpv_occupancy/frames')
    parser.add_argument('--frame-idx', type=int, default=[4509], nargs='+',
                        help='idx of frame to visualize, the idx corresponds to the order in pkl file.')
    parser.add_argument('--mode', type=int, default=0,
                        help='0: occupancy, 1: predicted point cloud, 2: gt point cloud')

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
    from nuscenes import NuScenes
    from visualization.dataset import ImagePoint_NuScenes_vis, DatasetWrapper_NuScenes_vis

    if args.vis_train:
        pkl_path = 'data/nuscenes_infos_train.pkl'
    else:
        pkl_path = 'data/nuscenes_infos_val.pkl'

    data_path = 'data/nuscenes'
    label_mapping = dataset_config['label_mapping']

    nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

    pt_dataset = ImagePoint_NuScenes_vis(
        data_path, imageset=pkl_path,
        label_mapping=label_mapping, nusc=nusc)

    dataset = DatasetWrapper_NuScenes_vis(
        pt_dataset,
        grid_size=cfg.grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["fill_label"],
        phase='val'
    )
    print(len(dataset))

    for index in args.frame_idx:
        print(f'processing frame {index}')
        # import pdb; pdb.set_trace()
        batch_data, filelist, scene_meta, timestamp = dataset[index]
        imgs, img_metas, vox_label, grid, pt_label = batch_data
        imgs_onnx = np.stack([imgs]).astype(np.float32)
        imgs = torch.from_numpy(np.stack([imgs]).astype(np.float32)).to(device)
        grid_onnx = np.stack([grid]).astype(np.float32)
        grid = torch.from_numpy(np.stack([grid]).astype(np.float32)).to(device)
        
        with torch.no_grad():
            # -------------------------------------------------------------
            # import pdb; pdb.set_trace()
            lidar2img = []
            img_shape = None
            new_img_metas = dict()
            for key in img_metas.keys():
                datas = img_metas[key]
                if key == 'lidar2img':
                    lidar2img.append(datas)
                if key == 'img_shape':
                    img_shape = datas
        
            # import pdb; pdb.set_trace()
            # ----------------------------------------------------------------------------------------
            lidar2img_onnx = np.array(lidar2img)
            lidar2img = torch.from_numpy(np.array(lidar2img)).to(device)
            img_shape_onnx = np.array([img_shape])
            img_shape = torch.from_numpy(np.array([img_shape])).to(device)
            

            # my_model.forward = my_model.forward_trt
            # torch.onnx.export(my_model, (imgs, grid.clone(), lidar2img, img_shape),
            #                   "/home/zs/Code/TPVFormer_new/tpvformer.onnx",
            #                   verbose=True,
            #                   opset_version=17,
            #                   input_names=["imgs", "points",
            #                                "lidar2img", "img_shape"],
            #                   output_names=["output1", "output2"],
            #                 #   dynamic_axes={"imgs": {0: "batch_size", 3: "img_h", 4: "img_w"}, "points": {0: "batch_size", 1: "num_points"}, "lidar2img": {
            #                 #       0: "batch_size"}, "img_shape": {0: "batch_size"}, "output1": {0: "batch_size"}, "output2": {0: "batch_size", 2: "num_points"}},
            #                   dynamic_axes={"imgs": {0: "batch_size", 3: "img_h", 4: "img_w"}, "points": {0: "batch_size", 1: "num_points"}, "lidar2img": {
            #                       0: "batch_size"}, "img_shape": {0: "batch_size"}},
            #                     # dynamic_axes={"points": {1: "num_points"}, "output2": {2: "num_points"}},
            #                   autograd_inlining=False)
            
            # print("TPVFormer Convert to ONNX Done!!!")
            onnx_path = "/home/zs/Code/TPVFormer_new/tpvformer_sim.onnx"
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            print("----------------------------------------")
            # ort_custom_op_path = get_onnxruntime_op_path()
            ort_custom_op_path = "/home/zs/Code/mmdeploy/build/lib/libmmdeploy_onnxruntime_ops.so"
            # import pdb; pdb.set_trace()
            assert os.path.exists(ort_custom_op_path)
            session_options = ort.SessionOptions()
            session_options.register_custom_ops_library(ort_custom_op_path)
            sess = ort.InferenceSession(onnx_path , session_options, providers=['CUDAExecutionProvider'])
            output_layers_name = ['output1', 'output2']
            onnx_output = sess.run(output_layers_name, {"imgs":imgs_onnx, "points":grid_onnx, "lidar2img":lidar2img_onnx, "img_shape":img_shape_onnx})
            print(onnx_output)
            print("-----------------------------")
            print(onnx_output[0][0][0])
            print("-------------------------")
            print(onnx_output[1][0])
            # sess = onnxruntime.InferenceSession(onnx_path)

            # onnx_output = sess.run(output_layers_name, (imgs, grid.clone(), lidar2img, img_shape))
            # import pdb; pdb.set_trace()