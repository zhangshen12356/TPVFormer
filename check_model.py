import sys
import onnx
filename = "/home/zs/Code/TPVFormer_new/tpvformer_sim.onnx"
model = onnx.load(filename)
onnx.checker.check_model(model)