import pickle
import pdb

pickle_path = "/home/zs/Code/TPVFormer_new/data/nuscenes_infos_val.pkl"
with open(pickle_path, 'rb') as file:
    data = pickle.load(file)  
    '''
    # data.keys():dict_keys(['infos', 'metadata'])
    data['metadata'] = {'version': 'v1.0-trainval'}
    data['infos']: 是一个列表,val的长度是6019
    
    data['infos'][0].keys():dict_keys(['lidar_path', 'token', 'sweeps', 'cams', 'lidar2ego_translation', 'lidar2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts', 'num_radar_pts'])
    
    data['infos'][0]['lidar_path']:'./data/nuscenes/samples/LIDAR_TOP/n015-2018-08-02-17-16-37+0800__LIDAR_TOP__1533201470448696.pcd.bin'
    data['infos'][0]['token']: 'fd8420396768425eabec9bdddf7e64b6'
    data['infos'][0]['sweeps']: []
    data['infos'][0]['cams'].keys():dict_keys(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'])
                                  data['infos'][0]['cams']['CAM_FRONT'].keys():dict_keys(['data_path', 'type', 'sample_data_token', 'sensor2ego_translation', 'sensor2ego_rotation', 'ego2global_translation', 'ego2global_rotation', 'timestamp', 'sensor2lidar_rotation', 'sensor2lidar_translation', 'cam_intrinsic'])
    
    data['infos'][0]['lidar2ego_translation']:[0.943713, 0.0, 1.84023]
    data['infos'][0]['lidar2ego_rotation']: [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817]
    data['infos'][0]['ego2global_translation']: [249.89610931430778, 917.5522573162784, 0.0]
    data['infos'][0]['ego2global_rotation']: [0.9984303573176436, -0.008635865272570774, 0.0025833156025800875, -0.05527720957189669]
    data['infos'][0]['timestamp']: 1533201470448696
    data['infos'][0]['gt_boxes']: shape (37, 7)
    data['infos'][0]['gt_names']:   array(['car', 'car', 'car', 'car', 'pedestrian', 'car', 'pedestrian',
                                        'car', 'pedestrian', 'pedestrian', 'traffic_cone', 'traffic_cone',
                                        'car', 'car', 'car', 'car', 'car', 'car', 'car', 'pedestrian',
                                        'car', 'car', 'car', 'car', 'car', 'car', 'pedestrian', 'car',
                                        'car', 'car', 'traffic_cone', 'car', 'pedestrian', 'pedestrian',
                                        'car', 'pedestrian', 'car'], dtype='<U12')
    data['infos'][0]['gt_velocity']: shape (37, 2)
    
    data['infos'][0]['num_lidar_pts']: array([169,  34, 475,   6,   2,   6,   1,   3,   2,   5,   2,   3,   2,
                                                1,   1,   8,   5,   1,   7,   2, 492,   2,  11,  54,   4,  21,
                                                2,   3,  19,   1,   1,   8,   2,   5, 119,   7,  55])
    data['infos'][0]['num_radar_pts']: array([4, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0,0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0])
    '''
    
    pdb.set_trace()