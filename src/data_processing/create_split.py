from glob import glob
import random
import numpy as np
import config.config_loader as cfg_loader
import argparse
import os
from data_processing import utils
from scipy.spatial import cKDTree as KDTree
import trimesh
import sys
sys.path.append('./')

parser = argparse.ArgumentParser(
    description='Generates a data split file.'
)

parser.add_argument('config', type=str, help='Path to config file.')

args = parser.parse_args()

cfg = cfg_loader.load(args.config)

# shorthands
bbox = cfg['data_bounding_box']
res = cfg['input_resolution']
num_points = cfg['input_points_number']
bbox_str = cfg['data_bounding_box_str']

grid_points = utils.create_grid_points_from_xyz_bounds(*bbox, res)
kdtree = KDTree(grid_points)


def get_partial_score(partial_paths, cfg):
    scores = []
    for partial_path in partial_paths:
        file_name = partial_path.split(os.sep)[-1][:-4]
        path_dir = os.path.dirname(partial_path)
        gt_file_name = path_dir.split(os.sep)[-1]
        file_type = path_dir.split(os.sep)[-2]
        full_path = os.path.join(
            cfg['data_path'], file_type[:-8], gt_file_name, gt_file_name + '_normalized.obj')

        full_mesh = utils.as_mesh(trimesh.load(full_path))
        partial_mesh = utils.as_mesh(trimesh.load(partial_path))

        partial_point_cloud = partial_mesh.sample(num_points)
        partial_occupancies = np.zeros(len(grid_points), dtype=np.int8)
        _, idx_partial = kdtree.query(partial_point_cloud)
        partial_occupancies[idx_partial] = 1

        full_point_cloud = full_mesh.sample(num_points)
        full_occupancies = np.zeros(len(grid_points), dtype=np.int8)
        _, idx_full = kdtree.query(full_point_cloud)
        full_occupancies[idx_full] = 1

        count_occ = full_occupancies == 1

        score = partial_occupancies == full_occupancies
        score = np.sum(score*count_occ) / np.sum(count_occ)
        scores.append(score)
        print(partial_path, score)

    return scores


if cfg['action'] == "texture":
    train_all = glob(os.path.join(cfg['data_path'], 'train_partial', cfg['preprocessing']
                     ['voxelized_colored_pointcloud_sampling']['input_files_regex'][3:]))
    random.shuffle(train_all)
    val = train_all[:int(len(train_all)*0.1)]
    train = train_all[int(len(train_all)*0.1):]
    test = glob(os.path.join(cfg['data_path'], 'test_partial', cfg['preprocessing']
                ['voxelized_colored_pointcloud_sampling']['input_files_regex'][3:]))
    predict = glob(os.path.join(cfg['data_path'], 'test-codalab-partial', cfg['preprocessing']
                   ['voxelized_colored_pointcloud_sampling']['input_files_regex'][3:]))


elif cfg['action'] == "geometry":
    train_all = glob(os.path.join(cfg['data_path'], 'train_partial', cfg['preprocessing']
                     ['voxelized_pointcloud_sampling']['input_files_regex'][3:]))
    random.shuffle(train_all)
    val = train_all[:int(len(train_all)*0.1)]
    train = train_all[int(len(train_all)*0.1):]
    test = glob(os.path.join(cfg['data_path'], 'test_partial', cfg['preprocessing']
                ['voxelized_pointcloud_sampling']['input_files_regex'][3:]))
    predict = glob(os.path.join(cfg['data_path'], 'val_partial',
                   cfg['preprocessing']['voxelized_pointcloud_sampling']['input_files_regex'][3:]))

train, val, test, predict = map(sorted, [train, val, test, predict])
train_score = get_partial_score(train, cfg)
val_score = get_partial_score(val, cfg)
test_score = get_partial_score(test, cfg)


split_dict = {'train': train, 'train_score': train_score,
              'test': test, 'test_score': test_score, 'val': val, 'val_score': val_score, 'predict': predict}

np.savez(cfg['split_file'], **split_dict)
