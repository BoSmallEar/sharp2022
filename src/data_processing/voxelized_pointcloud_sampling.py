import data_processing.utils as utils
import tqdm
import traceback
import config.config_loader as cfg_loader
from scipy.spatial import cKDTree as KDTree
import numpy as np
import trimesh
from glob import glob
import os
import multiprocessing as mp
from multiprocessing import Pool
import argparse

PATH_TO_YOUR_SMPL_MODEL = ''


def voxelized_pointcloud_sampling(path):
    try:
        path = os.path.normpath(path)
        gt_file_name = path.split(os.sep)[-2]
        full_file_name = path.split(os.sep)[-1][:-4]
        split_name = path.split(os.sep)[-3]

        if not args.smpl:
            out_file = os.path.dirname(path) + '/{}_voxelized_point_cloud_res{}_points{}_bbox{}.npz'\
                .format(full_file_name, res, num_points, bbox_str)
        else:
            if split_name == "train_partial":
                split_name = "train_smpl"
            elif split_name == "test_partial":
                split_name = "test_smpl"

            smpl_file_name = f"{gt_file_name}_pose_smpl_model.obj"

            smpl_path = os.path.join(
                PATH_TO_YOUR_SMPL_MODEL, gt_file_name, full_file_name + '_pose_smpl_model.obj')
            out_file = os.path.dirname(path) + '/{}_voxelized_point_cloud_res{}_points{}_bbox{}_smpl_estimated.npz'\
                .format(full_file_name, res, num_points, bbox_str)

        if os.path.exists(out_file):
            print('File exists. Done.')
            return

        mesh = utils.as_mesh(trimesh.load(path))
        point_cloud = mesh.sample(num_points)

        occupancies = np.zeros(len(grid_points), dtype=np.int8)

        _, idx = kdtree.query(point_cloud)
        occupancies[idx] = 1

        compressed_occupancies = np.packbits(occupancies)

        if args.smpl:
            smpl_mesh = utils.as_mesh(trimesh.load(smpl_path))
            smpl_point_cloud = mesh.sample(num_points)
            smpl_occupancies = np.zeros(len(grid_points), dtype=np.int8)
            _, idx = kdtree.query(smpl_point_cloud)
            occupancies[idx] = 1
            smpl_compressed_occupancies = np.packbits(smpl_occupancies)
            np.savez(out_file, point_cloud=point_cloud, compressed_occupancies=compressed_occupancies, res=res,
                     smpl_point_cloud=smpl_point_cloud, smpl_compressed_occupancies=smpl_compressed_occupancies)
        else:
            np.savez(out_file, point_cloud=point_cloud,
                     compressed_occupancies=compressed_occupancies, res=res)
        print('Finished {}'.format(path))

    except Exception as err:
        print('Error with {}: {}'.format(path, traceback.format_exc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generates the input for the network: a partial colored shape and a uncolorized, but completed shape. \
        Both encoded as 3D voxel grids for usage with a 3D CNN.'
    )

    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--smpl', action="store_true")
    args = parser.parse_args()

    cfg = cfg_loader.load(args.config)

    # shorthands
    bbox = cfg['data_bounding_box']
    res = cfg['input_resolution']
    num_points = cfg['input_points_number']
    bbox_str = cfg['data_bounding_box_str']

    grid_points = utils.create_grid_points_from_xyz_bounds(*bbox, res)
    kdtree = KDTree(grid_points)

    print('Fining all input partial paths for voxelization.')
    paths = glob(cfg['data_path'] + cfg['preprocessing']
                 ['voxelized_pointcloud_sampling']['input_files_regex'])

    print('Start voxelization.')
    p = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(p.imap_unordered(voxelized_pointcloud_sampling, paths), total=len(paths)):
        pass
    p.close()
    p.join()
