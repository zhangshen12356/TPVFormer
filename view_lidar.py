import numpy as np
import mayavi.mlab
 
# lidar_path换成自己的.bin文件路径
pointcloud = np.fromfile(str("./data/nuscenes/lidarseg/v1.0-trainval/84ad449e90a1432093254dfd5eea8e35_lidarseg.bin"), dtype=np.float32, count=-1)
import pdb; pdb.set_trace()
