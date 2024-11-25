import onnx
import onnx_graphsurgeon as gs

# graph = gs.import_onnx(onnx.load("/home/zs/Code/TPVFormer_new/tpvformer_encoder_sim.onnx"))
# graph = gs.import_onnx(onnx.load("/home/zs/Code/TPVFormer_new/tpvformer_sim.onnx"))
graph = gs.import_onnx(onnx.load("/home/zs/Code/TPVFormer_new/tpvformer_no_points_sim.onnx"))

for node in graph.nodes:
    print(node.op)
    if node.op == "Reshape":
        node.attrs["allowzero"] = 1

# onnx.save(gs.export_onnx(graph), "/home/zs/Code/TPVFormer_new/tpvformer_sim_change.onnx")
onnx.save(gs.export_onnx(graph), "tpvformer_no_points_sim_change.onnx")