import pycuda.autoinit
import tensorrt as trt
import numpy as np
import cv2
import argparse
import os
import torch
import mmcv
from mmcv import Config
import onnx
from collections import OrderedDict
from typing import Union, Optional, Sequence, Dict, Any
import ctypes
from tensorrt import get_plugin_registry


def revise_ckpt(state_dict):
    tmp_k = list(state_dict.keys())[0]
    if tmp_k.startswith('module.'):
        state_dict = OrderedDict(
            {k[7:]: v for k, v in state_dict.items()})
    return state_dict


class TRTWrapper(torch.nn.Module):
    def __init__(self, engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        lib_path = '/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so'
        custom_layer_lib = ctypes.CDLL(lib_path)
        # plugin_registry = get_plugin_registry()
        # import pdb; pdb.set_trace()
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        # import pdb; pdb.set_trace()
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0
        # import pdb; pdb.set_trace()
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)
            # import pdb; pdb.set_trace()
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, 'Input shape should be between ' + \
                    f'{profile[0]} and {profile[2]}' + \
                    f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs


def creat_engine(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    lib_path = '/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so'
    custom_layer_lib = ctypes.CDLL(lib_path)
    onnx_model = onnx.load(onnx_path)
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30)*42  # 42G
    profile = builder.create_optimization_profile()

    # profile.set_shape('imgs', [6, 3, 928, 1600], [
    #                   6, 3, 928, 1600], [6, 3, 928, 1600])
    profile.set_shape('imgs', [1, 6, 3, 928, 1600], [
                      1, 6, 3, 928, 1600], [1, 6, 3, 928, 1600])
    profile.set_shape('points', [1, 1000, 3], [1, 34752, 3], [1, 50000, 3])
    profile.set_shape('lidar2img', [1, 6, 4, 4], [1, 6, 4, 4], [1, 6, 4, 4])
    profile.set_shape('img_shape', [1, 6, 3], [1, 6, 3], [1, 6, 3])

    config.add_optimization_profile(profile)
    device = torch.device('cuda:0')
    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open('/home/zs/Code/TPVFormer_new/tpvformer_sim_change.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!!!!!")

def creat_aggregator_engine(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    lib_path = '/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so'
    custom_layer_lib = ctypes.CDLL(lib_path)
    onnx_model = onnx.load(onnx_path)
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)
    # parser.register_custom_op_library(custom_layer_lib)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30)*42  # 42G
    profile = builder.create_optimization_profile()

    profile.set_shape('input1', [1, 10000, 256], [1, 10000, 256], [1, 10000, 256])
    profile.set_shape('input2', [1, 800, 256], [1, 800, 256], [1, 800, 256])
    profile.set_shape('input3', [1, 800, 256], [1, 800, 256], [1, 800, 256])
    profile.set_shape('input4', [1, 10000, 3], [1, 34752, 3], [1, 50000, 3])

    config.add_optimization_profile(profile)
    device = torch.device('cuda:0')
    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open('/home/zs/Code/TPVFormer_new/tpvformer_aggregator_sim_change.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")
        
def creat_engine_self_attention(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    lib_path = '/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so'
    custom_layer_lib = ctypes.CDLL(lib_path)
    onnx_model = onnx.load(onnx_path)
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30)*42  # 42G
    profile = builder.create_optimization_profile()

    profile.set_shape('input1', [1, 10000, 256], [
                      1, 10000, 256], [1, 10000, 256])
    profile.set_shape('input2', [1, 10000, 256], [
                      1, 10000, 256], [1, 10000, 256])
    profile.set_shape('input3', [2, 10000, 1, 2], [
                      2, 10000, 1, 2], [2, 10000, 1, 2])
    profile.set_shape('input4', [1, 2], [1, 2], [1, 2])
    # profile.set_shape('input5', [1], [1], [1])

    config.add_optimization_profile(profile)
    device = torch.device('cuda:0')
    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open('/home/zs/Code/TPVFormer_new/tpvformer_self_attention_sim_change.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")

def creat_engine_cross_attention(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    lib_path = '/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so'
    custom_layer_lib = ctypes.CDLL(lib_path)
    onnx_model = onnx.load(onnx_path)
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30)*42  # 42G
    profile = builder.create_optimization_profile()

    profile.set_shape('input1', [1, 11600, 256],[1, 11600, 256],[1, 11600, 256])
    profile.set_shape('input2', [6, 30825, 1, 256],[6, 30825, 1, 256],[6, 30825, 1, 256])
    profile.set_shape('input3', [6, 30825, 1, 256], [6, 30825, 1, 256], [6, 30825, 1, 256])
    profile.set_shape('input4', [6, 1, 10000, 4, 2], [6, 1, 10000, 4, 2], [6, 1, 10000, 4, 2])
    profile.set_shape('input5', [6, 1, 800, 32, 2], [6, 1, 800, 32, 2], [6, 1, 800, 32, 2])
    profile.set_shape('input6', [6, 1, 800, 32, 2], [6, 1, 800, 32, 2], [6, 1, 800, 32, 2])
    profile.set_shape('input7', [6, 1, 10000, 4], [6, 1, 10000, 4], [6, 1, 10000, 4])
    profile.set_shape('input8', [6, 1, 800, 32], [6, 1, 800, 32], [6, 1, 800, 32])
    profile.set_shape('input9', [6, 1, 800, 32], [6, 1, 800, 32], [6, 1, 800, 32])
    profile.set_shape('input10', [4, 2], [4, 2], [4, 2])
    profile.set_shape('input11', [4], [4], [4])

    config.add_optimization_profile(profile)
    device = torch.device('cuda:0')
    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open('/home/zs/Code/TPVFormer_new/tpvformer_cross_attention_sim_change.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")
        
def creat_engine_neck(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    lib_path = '/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so'
    custom_layer_lib = ctypes.CDLL(lib_path)
    onnx_model = onnx.load(onnx_path)
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30)*42  # 42G
    profile = builder.create_optimization_profile()

    profile.set_shape('input1', [6, 512, 116, 200],[6, 512, 116, 200],[6, 512, 116, 200])
    profile.set_shape('input2', [6, 1024, 58, 100],[6, 1024, 58, 100],[6, 1024, 58, 100])
    profile.set_shape('input3', [6, 2048, 29, 50], [6, 2048, 29, 50], [6, 2048, 29, 50])

    config.add_optimization_profile(profile)
    device = torch.device('cuda:0')
    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open('/home/zs/Code/TPVFormer_new/tpvformer_neck_sim_change.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")

def creat_engine_head(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    lib_path = '/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so'
    custom_layer_lib = ctypes.CDLL(lib_path)
    onnx_model = onnx.load(onnx_path)
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30)*42  # 42G
    profile = builder.create_optimization_profile()

    profile.set_shape('input1', [1, 6, 256, 116, 200],[1, 6, 256, 116, 200],[1, 6, 256, 116, 200])
    profile.set_shape('input2', [1, 6, 256, 58, 100],[1, 6, 256, 58, 100],[1, 6, 256, 58, 100])
    profile.set_shape('input3', [1, 6, 256, 29, 50], [1, 6, 256, 29, 50], [1, 6, 256, 29, 50])
    profile.set_shape('input4', [1, 6, 256, 15, 25], [1, 6, 256, 15, 25], [1, 6, 256, 15, 25])
    profile.set_shape('input5', [1, 6, 4, 4], [1, 6, 4, 4], [1, 6, 4, 4])
    profile.set_shape('input6', [1, 6, 3], [1, 6, 3], [1, 6, 3])

    config.add_optimization_profile(profile)
    device = torch.device('cuda:0')
    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open('/home/zs/Code/TPVFormer_new/tpvformer_head_sim_change.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")

def creat_engine_encoder(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    lib_path = '/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so'
    custom_layer_lib = ctypes.CDLL(lib_path)
    onnx_model = onnx.load(onnx_path)
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30)*42  # 42G
    profile = builder.create_optimization_profile()

    profile.set_shape('input1', [1, 10000, 256],[1, 10000, 256],[1, 10000, 256])
    profile.set_shape('input2', [1, 800, 256],[1, 800, 256],[1, 800, 256])
    profile.set_shape('input3', [1, 800, 256], [1, 800, 256], [1, 800, 256])
    profile.set_shape('input4', [6, 30825, 1, 256], [6, 30825, 1, 256], [6, 500000, 1, 256])  # !!!
    profile.set_shape('input5', [6, 30825, 1, 256], [6, 30825, 1, 256], [6, 500000, 1, 256])  # !!!
  
    profile.set_shape('input9', [1, 10000, 256], [1, 10000, 256], [1, 10000, 256])
    profile.set_shape('input10', [4, 2], [4, 2], [4, 2]) # !!!

    profile.set_shape('input12', [1, 6, 4, 4], [1, 6, 4, 4], [1, 6, 4, 4])  # !!!
    profile.set_shape('input13', [1, 6, 3], [1, 6, 3], [1, 6, 3])  # !!!

    config.add_optimization_profile(profile)
    device = torch.device('cuda:0')
    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open('/home/zs/Code/TPVFormer_new/tpvformer_encoder_sim_change.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")

def creat_engine_layer(onnx_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    lib_path = '/home/zs/Code/TPVFormer_TRT/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so'
    custom_layer_lib = ctypes.CDLL(lib_path)
    onnx_model = onnx.load(onnx_path)
    # create builder and network
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    EXPLICIT_BATCH = 1 << (int)(
        trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    # parse onnx
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_model.SerializeToString()):
        error_msgs = ''
        for error in range(parser.num_errors):
            error_msgs += f'{parser.get_error(error)}\n'
        raise RuntimeError(f'Failed to parse onnx, {error_msgs}')

    config = builder.create_builder_config()
    config.max_workspace_size = (1 << 30)*42  # 42G
    profile = builder.create_optimization_profile()

    profile.set_shape('input1', [1, 10000, 256],[1, 10000, 256],[1, 10000, 256])
    profile.set_shape('input2', [1, 800, 256],[1, 800, 256],[1, 800, 256])
    profile.set_shape('input3', [1, 800, 256], [1, 800, 256], [1, 800, 256])
    profile.set_shape('input4', [6, 30825, 1, 256], [6, 30825, 1, 256], [6, 30825, 1, 256])
    profile.set_shape('input5', [6, 30825, 1, 256], [6, 30825, 1, 256], [6, 30825, 1, 256])
    profile.set_shape('input6', [1, 10000, 256], [1, 10000, 256], [1, 10000, 256])
    profile.set_shape('input7', [2, 10000, 1, 2], [2, 10000, 1, 2], [2, 10000, 1, 2])
    # profile.set_shape('input8', [100], [100], [100])
    # profile.set_shape('input9', [100], [100], [100])
    # profile.set_shape('input10', [8], [8], [8])
    profile.set_shape('input11', [6, 1, 10000, 4, 2], [6, 1, 10000, 4, 2], [6, 1, 10000, 4, 2])
    profile.set_shape('input12', [6, 1, 800, 32, 2], [6, 1, 800, 32, 2], [6, 1, 800, 32, 2])
    profile.set_shape('input13', [6, 1, 800, 32, 2], [6, 1, 800, 32, 2], [6, 1, 800, 32, 2])
    profile.set_shape('input14', [6, 1, 10000, 4], [6, 1, 10000, 4], [6, 1, 10000, 4])
    profile.set_shape('input15', [6, 1, 800, 32], [6, 1, 800, 32], [6, 1, 800, 32])
    profile.set_shape('input16', [6, 1, 800, 32], [6, 1, 800, 32], [6, 1, 800, 32])
    profile.set_shape('input17', [4, 2], [4, 2], [4, 2])
    # profile.set_shape('input18', [4], [4], [4])

    config.add_optimization_profile(profile)
    device = torch.device('cuda:0')
    # create engine
    with torch.cuda.device(device):
        engine = builder.build_engine(network, config)

    with open('/home/zs/Code/TPVFormer_new/tpvformer_layers_sim_change.engine', mode='wb') as f:
        f.write(bytearray(engine.serialize()))
        print("generating file done!")
        
def infer_engine(engine_file):
    import sys
    sys.path.insert(0, os.path.abspath('.'))

    device = torch.device('cuda:0')

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv04_occupancy_trt.py')
    parser.add_argument('--work-dir', type=str, default='out/tpv_occupancy')
    parser.add_argument('--ckpt-path', type=str,
                        default='/home/zs/Code/TPVFormer_new/ckpts/tpv04_occupancy_v2.pth')
    parser.add_argument('--vis-train', action='store_true', default=False)
    parser.add_argument('--save-path', type=str,
                        default='out/tpv_occupancy/frames')
    parser.add_argument('--frame-idx', type=int, default=4509, nargs='+',
                        help='idx of frame to visualize, the idx corresponds to the order in pkl file.')
    parser.add_argument('--mode', type=int, default=0,
                        help='0: occupancy, 1: predicted point cloud, 2: gt point cloud')

    args = parser.parse_args()
    # print(args)

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
        # print(my_model.load_state_dict(revise_ckpt(ckpt)))
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
    # print(len(dataset))

    batch_data, filelist, scene_meta, timestamp = dataset[args.frame_idx]
    imgs, img_metas, vox_label, grid, pt_label = batch_data
    imgs = torch.from_numpy(np.stack([imgs]).astype(np.float32)).to(device)
    grid = torch.from_numpy(np.stack([grid]).astype(np.float32)).to(device)
    lidar2img = []
    img_shape = None
    new_img_metas = dict()
    for key in img_metas.keys():
        datas = img_metas[key]
        if key == 'lidar2img':
            lidar2img.append(datas)
        if key == 'img_shape':
            img_shape = datas
    lidar2img = torch.from_numpy(np.array(lidar2img)).to(device)
    img_shape = torch.from_numpy(np.array([img_shape])).to(device)
    model = TRTWrapper(engine_file, ['output1', 'output2'])
    output = model({"imgs": imgs, "points": grid,
                   "lidar2img": lidar2img, "img_shape": img_shape})
    # output = model({"imgs": imgs, "points": grid})
    import pdb; pdb.set_trace()
    print(output)


def infer_engine_self_attention(engine_file):
    device = torch.device('cuda:0')
    model = TRTWrapper(engine_file, ['output1'])
    input1 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input1.npy")).to(device)
    input2 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input2.npy")).to(device)
    input3 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input3.npy")).to(device)
    input4 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input4.npy")).to(device)
    output = model({"input1": input1, "input2": input2,
                   "input3": input3, "input4": input4})
    import pdb; pdb.set_trace()
    print(output)

def infer_engine_cross_attention(engine_file):
    device = torch.device('cuda:0')
    model = TRTWrapper(engine_file, ['output1'])
    input1 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input1.npy")).to(device)
    input2 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input2.npy")).to(device)
    input3 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input3.npy")).to(device)
    input4 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input4.npy")).to(device)
    input5 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input5.npy")).to(device)
    input6 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input6.npy")).to(device)
    input7 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input7.npy")).to(device)
    input8 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input8.npy")).to(device)
    input9 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input9.npy")).to(device)
    input10 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input10.npy")).to(device)
    input11 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/cross_atten_input11.npy")).to(device)
    import pdb; pdb.set_trace()
    output = model({"input1": input1, "input2": input2,
                   "input3": input3, "input4": input4, "input5":input5, "input6":input6, "input7":input7, "input8":input8, "input9":input9, "input10":input10})
    import pdb; pdb.set_trace()
    print(output)


def infer_engine_aggregator(engine_file):
    device = torch.device('cuda:0')
    model = TRTWrapper(engine_file, ['output1', 'output2'])
    input1 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/aggregator_input1.npy")).to(device)
    input2 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/aggregator_input2.npy")).to(device)
    input3 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/aggregator_input3.npy")).to(device)
    input4 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/aggregator_input4.npy")).to(device)
    
    output = model({"input1": input1, "input2": input2,
                   "input3": input3, "input4": input4})
    # import pdb; pdb.set_trace()
    print(output['output1'][0][0][0][0])
    
def infer_engine_neck(engine_file):
    device = torch.device('cuda:0')
    model = TRTWrapper(engine_file, ['output1', 'output2', 'output3', 'output4'])
    input1 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/neck_input1.npy")).to(device)
    input2 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/neck_input2.npy")).to(device)
    input3 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/neck_input3.npy")).to(device)
    
    output = model({"input1": input1, "input2": input2,
                   "input3": input3})
    # import pdb; pdb.set_trace()
    print(output['output1'][0][0])

def infer_engine_head(engine_file):
    device = torch.device('cuda:0')
    model = TRTWrapper(engine_file, ['output1', 'output2', 'output3'])
    input1 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/head_input1.npy")).to(device)
    input2 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/head_input2.npy")).to(device)
    input3 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/head_input3.npy")).to(device)
    input4 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/head_input4.npy")).to(device)
    input5 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/head_input5.npy")).to(device)
    input6 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/head_input6.npy")).to(device)
    
    output = model({"input1": input1, "input2": input2,
                   "input3": input3, "input4": input4, "input5": input5, "input6": input6})
    import pdb; pdb.set_trace()
    print(output['output1'][0][0])

def infer_engine_encoder(engine_file):
    device = torch.device('cuda:0')
    # model = TRTWrapper(engine_file, ['output1', 'output2', 'output3'])
    model = TRTWrapper(engine_file, ['output1'])
    input1 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input1.npy")).to(device)
    input2 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input2.npy")).to(device)
    input3 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input3.npy")).to(device)
    input4 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input4.npy")).to(device)
    input5 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input5.npy")).to(device)
    # input6 = torch.from_numpy(np.array([np.load("/home/zs/Code/TPVFormer_new/encoder_input6.npy").tolist()])).to(device)
    # input7 = torch.from_numpy(np.array([np.load("/home/zs/Code/TPVFormer_new/encoder_input7.npy").tolist()])).to(device)
    # input8 = torch.from_numpy(np.array([np.load("/home/zs/Code/TPVFormer_new/encoder_input8.npy").tolist()])).to(device)
    input9 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input9.npy")).to(device)
    input10 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input10.npy")).to(device)
    input11 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input11.npy")).to(device)
    input12 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input12.npy")).to(device)
    input13 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input13.npy")).to(device)
    
    # output = model({"input1": input1, "input2": input2,
    #                "input3": input3,"input9":input9})
    # output = model({"input12":input12, "input13":input13})
    # output = model({"input12":input12})
    # output = model({"input1":input1, "input2":input2, "input3":input3})
    # output = model({"input13":input13})
    # output = model({"input13":input13})
    # output = model({"input4":input4})
    output = model({"input1":input1})
    # output = model({"input1": input1, "input2": input2,
    #                "input3": input3, "input4": input4, "input5": input5, "input9":input9, "input10":input10, "input12":input12, "input13":input13})
    import pdb; pdb.set_trace()
    print(output['output1'][0][0])


def infer_engine_layers(engine_file):
    device = torch.device('cuda:0')
    model = TRTWrapper(engine_file, ['output1', 'output2', 'output3'])
    input1 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input1.npy")).to(device)
    input2 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input2.npy")).to(device)
    input3 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input3.npy")).to(device)
    input4 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input4.npy")).to(device)
    input5 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input5.npy")).to(device)
    # input6 = torch.from_numpy(np.array([np.load("/home/zs/Code/TPVFormer_new/encoder_input6.npy").tolist()])).to(device)
    # input7 = torch.from_numpy(np.array([np.load("/home/zs/Code/TPVFormer_new/encoder_input7.npy").tolist()])).to(device)
    # input8 = torch.from_numpy(np.array([np.load("/home/zs/Code/TPVFormer_new/encoder_input8.npy").tolist()])).to(device)
    input6 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input6.npy")).to(device)
    input7 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input7.npy")).to(device)
    # input8 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input10.npy")).to(device)
    # input9 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input11.npy")).to(device)
    # input10 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/encoder_input12.npy")).to(device)
    input11 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input11.npy")).to(device)
    input12 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input12.npy")).to(device)
    input13 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input13.npy")).to(device)
    input14 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input14.npy")).to(device)
    input15 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input15.npy")).to(device)
    input16 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input16.npy")).to(device)
    input17 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input17.npy")).to(device)
    input18 = torch.from_numpy(np.load("/home/zs/Code/TPVFormer_new/self_atten_input18.npy")).to(device)
    
    output = model({"input1": input1, "input2": input2,
                   "input3": input3, "input4": input4, "input5": input5, "input6":input6, "input7":input7, "input11":input11, "input12":input12, "input13":input13, "input14": input14, "input15":input15, "input16":input16, "input17":input17})
    import pdb; pdb.set_trace()
    print(output['output1'][0][0])
    
if __name__ == "__main__":
    creat_engine("/home/zs/Code/TPVFormer_new/tpvformer_sim_change.onnx")
    # creat_engine_self_attention('/home/zs/Code/TPVFormer_new/tpvformer_self_attention_sim_change.onnx')
    # creat_engine_cross_attention('/home/zs/Code/TPVFormer_new/tpvformer_cross_attention_sim_change.onnx')
    # creat_aggregator_engine('/home/zs/Code/TPVFormer_new/tpvformer_aggregator_sim_change.onnx')
    # creat_engine_neck('/home/zs/Code/TPVFormer_new/tpvformer_neck_sim_change.onnx')
    # creat_engine_head('/home/zs/Code/TPVFormer_new/tpvformer_head_sim_change.onnx')
    # creat_engine_encoder('/home/zs/Code/TPVFormer_new/tpvformer_encoder_sim_change.onnx')  # !!!
    # creat_engine_layer('/home/zs/Code/TPVFormer_new/tpvformer_layers_sim_change.onnx')
    infer_engine('/home/zs/Code/TPVFormer_new/tpvformer_sim_change.engine')
    # infer_engine_self_attention('/home/zs/Code/TPVFormer_new/tpvformer_self_attention_sim_change.engine')
    # infer_engine_cross_attention('/home/zs/Code/TPVFormer_new/tpvformer_cross_attention_sim_change.engine')
    # infer_engine_aggregator("/home/zs/Code/TPVFormer_new/tpvformer_aggregator_sim_change.engine")
    # infer_engine_neck('/home/zs/Code/TPVFormer_new/tpvformer_neck_sim_change.engine')
    # infer_engine_head('/home/zs/Code/TPVFormer_new/tpvformer_head_sim_change.engine')
    # infer_engine_encoder('/home/zs/Code/TPVFormer_new/tpvformer_encoder_sim_change.engine')
    # infer_engine_layers('/home/zs/Code/TPVFormer_new/tpvformer_layers_sim_change.engine')
    
    
    # #  export LD_LIBRARY_PATH=/media/data_12T/zhangshen/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib
