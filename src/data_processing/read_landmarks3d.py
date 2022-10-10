import numpy as np
from data_processing.utils import to_grid_sample_coords

def read(path, bbox, joint_num = 25):
    landmarks3d_list = []
    with open(path) as f:
        landmarks3d_txt = f.read().split('\n')
        for i in range(joint_num):
            landmarks3d = landmarks3d_txt[i].split(' ')
            landmarks3d_list.append(float(landmarks3d[1]))
            landmarks3d_list.append(float(landmarks3d[2]))
            landmarks3d_list.append(float(landmarks3d[3]))
    return to_grid_sample_coords(np.array(landmarks3d_list).reshape([joint_num,3]), bbox)
