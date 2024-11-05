import io
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
import random
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
import os
from lib.utils.linemod import linemod_config
import torch
if cfg.test.icp:
    from lib.utils import icp_utils
from PIL import Image
from lib.utils.img_utils import read_depth
from scipy import spatial
from scipy.spatial import distance
from scipy.spatial.transform import Rotation

import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class Evaluator:

    def __init__(self, result_dir):
        self.result_dir = result_dir
        dataset_log = DatasetCatalog()
        args = dataset_log.get(name = cfg.test.dataset)
        # args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

        data_root = args['data_root']
        cls = cfg.cls_type
        # should be consistent with cls_alpha_zoffset_map in stable_pose_est.py
        cls_str_to_int_map = {
            "mainshell": 0,
            "topshell": 1,
            "insert_mold": 2
        }
        self.cls_int = cls_str_to_int_map[cls]
        model_path = os.path.join('data', cls+'_train', 'model.ply')
        self.model = pvnet_data_utils.get_ply_model(model_path)
        # self.diameter = np.loadtxt('data/custom/diameter.txt').item()

        self.T_gt = []
        self.T_pre = []

        self.avg_2d_kpts_error = []
        self.proj2d = []
        self.add = []
        self.icp_add = []
        self.cmd5 = []
        self.mask_ap = []
        # self.euler_err = [] # degree
        self.angular_rotation_err = [] # degree
        # self.angular_quaternion_err = [] # degree
        self.trans_err = [] # meter
        self.icp_render = icp_utils.SynRenderer(cfg.cls_type) if cfg.test.icp else None

        self.experiment_names = []
        self.ablation_study_pose_list = dict()
        # key: String - experiment_name
        # value: pose_list[nx2x3x4], pose_list[i][0] is ground truth pose, pose_list[i][1] is calculated pose

        self.valid_data_cnt = 0

        self.max_euclidean_dis_bewteen_kpts = -np.inf
        self.min_euclidean_dis_bewteen_kpts = np.inf

    def calculate_avg_2d_kpts_error_in_pixels(self, kpt_pred, kpt_gt):
        error = np.linalg.norm(kpt_pred - kpt_gt) / np.sqrt(len(kpt_pred))
        self.avg_2d_kpts_error.append(error)

    def add_to_eval_list(self, pose_pre, pose_gt):
        self.T_gt.append(pose_gt)
        self.T_pre.append(pose_pre)

        t_pre = pose_pre[:, 3]
        t_gt = pose_gt[:, 3]
        translation_error = np.abs(t_pre - t_gt)
        # if(translation_error[2] > 10): # outliers, error > 100m
        #     return
        
        R_pre_in_world = pose_pre[:, :3]
        R_gt_in_world = pose_gt[:, :3]
        R_pre_2_gt = np.dot(R_pre_in_world, R_gt_in_world.T)
        trace = np.trace(R_pre_2_gt)
        trace = trace if trace <= 3 else 3
        trace = trace if trace >= -1 else -1
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))

        if(angular_distance > 10): #if theta>10 deg, consider is as an outlier and exclude it
            return
        
        self.valid_data_cnt += 1
        self.trans_err.append(translation_error)
        self.angular_rotation_err.append(angular_distance)

    def projection_2d(self, pose_pred, pose_targets, K, threshold=5):
        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

        self.proj2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=1):
        # diameter = self.diameter * percentage
        diameter = 10 / 1000
    
        model_pred = np.dot(self.model, pose_pred[:, :3].T) + pose_pred[:, 3]
        model_targets = np.dot(self.model, pose_targets[:, :3].T) + pose_targets[:, 3]

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        if icp:
            self.icp_add.append(mean_dist < diameter)
        else:
            self.add.append(mean_dist < diameter)

    def quaternion_angular_err(self, T_pre, T_gt):
        R1 = T_pre[:, :3]
        R2 = T_gt[:, :3]

        # Convert rotation matrices to quaternion representations
        Q1 = Rotation.from_matrix(R1).as_quat()
        Q2 = Rotation.from_matrix(R2).as_quat()

        # Compute the dot product between the quaternions
        def dot_product(q1, q2):
            return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
        Q_diff = dot_product(Q1, Q2)

        angular_diff = 2 * np.arccos(Q_diff)
        # Convert angular difference to degrees
        angular_diff_deg = np.rad2deg(angular_diff)
        # self.angular_quaternion_err.append(angular_diff_deg)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        trace = trace if trace >= -1 else -1
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.cmd5.append(translation_distance < 5 and angular_distance < 5)

    def mask_iou(self, output, batch):
        mask_pred = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        mask_gt = batch['mask'][0].detach().cpu().numpy()
        iou = (mask_pred & mask_gt).sum() / (mask_pred | mask_gt).sum()
        self.mask_ap.append(iou > 0.9)

    def icp_refine(self, pose_pred, anno, output, K):
        depth = read_depth(anno['depth_path'])
        mask = torch.argmax(output['seg'], dim=1)[0].detach().cpu().numpy()
        if pose_pred[2, 3] <= 0 or np.sum(mask) < 20:
            return pose_pred
        depth[mask != 1] = 0
        pose_pred_tmp = pose_pred.copy()
        pose_pred_tmp[:3, 3] = pose_pred_tmp[:3, 3] * 1000
        R_refined, t_refined = icp_utils.icp_refinement(depth, self.icp_render, pose_pred_tmp[:3, :3], pose_pred_tmp[:3, 3], K.copy(), (depth.shape[1], depth.shape[0]), depth_only=True,            max_mean_dist_factor=5.0)
        R_refined, _ = icp_utils.icp_refinement(depth, self.icp_render, R_refined, t_refined, K.copy(), (depth.shape[1], depth.shape[0]), no_depth=True)
        pose_pred = np.hstack((R_refined, t_refined.reshape((3, 1)) / 1000))
        return pose_pred

    def get_top4_vectors_indices(self, var):
        # Compute the trace of each covariance matrix
        traces = np.trace(var, axis1=-2, axis2=-1)
        # Get the indices that would sort the traces in ascending order
        sorted_indices = np.argsort(traces)

        # Select the top 4 indices
        top_4_indices = sorted_indices[:4]

        return top_4_indices

    def add_to_ablation_study_pose_list(self, experiment_name, pose_pred, pose_gt):
        if experiment_name in self.ablation_study_pose_list:
            self.ablation_study_pose_list[experiment_name].append([pose_gt, pose_pred])
        else:
            self.ablation_study_pose_list[experiment_name] = [[pose_gt, pose_pred]]

    def evaluate(self, output, batch):
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])

        # top4_kpts_index = self.get_top4_vectors_indices(output['var'][0].detach().cpu().numpy())
        # kpt_3d = kpt_3d[top4_kpts_index]
        # kpt_2d = kpt_2d[top4_kpts_index]

        # 1. Raw output from PvNet
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        if self.icp_render is not None:
            pose_pred_icp = self.icp_refine(pose_pred.copy(), anno, output, K)
            self.add_metric(pose_pred_icp, pose_gt, icp=True)
        self.projection_2d(pose_pred, pose_gt, K)
        if cfg.cls_type in ['eggbox', 'glue']:
            self.add_metric(pose_pred, pose_gt, syn=True)
        else:
            self.add_metric(pose_pred, pose_gt)
        self.cm_degree_5_metric(pose_pred, pose_gt)
        self.mask_iou(output, batch)
        self.add_to_eval_list(pose_pred, pose_gt)
        self.quaternion_angular_err(pose_pred, pose_gt)
        kpt_pred = output['kpt_2d'].squeeze().cpu().numpy()
        kpt_gt = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)
        self.calculate_avg_2d_kpts_error_in_pixels(kpt_pred, kpt_gt) 

        # 2. Raw output from PvNet with orientation refinement
        # Pose Refinement: need T_cam_in_world, T_object_in_world
        if cfg.test.enable_refinement :
            from mrcnn.stable_poses_est import construct_T_stable_from_T_est
            T_cam_in_world = np.array(anno['cam_pose_world']) # 4x4
            # T_object_gt_in_world = np.array(anno['object_pose_world']) # 4x4
            # T_object_in_cam = np.linalg.inv(T_cam_in_world) @ T_object_in_world # T_object_in_cam[:3][:]== pose_gt
            T_pred_in_cam = np.vstack([pose_pred, [0, 0, 0, 1]])
            T_pred_in_world =  T_cam_in_world @ T_pred_in_cam
            T_stable_in_world = construct_T_stable_from_T_est(T_pred_in_world, self.cls_int)
            T_stable_in_cam = np.linalg.inv(T_cam_in_world) @ T_stable_in_world
            # Replace back x and y, stable-pose refinement is only for z and orientation
            T_stable_in_cam[0][3] = T_pred_in_cam[0][3]
            T_stable_in_cam[1][3] = T_pred_in_cam[1][3]
            refinement_experiment_name = "pose estimation with stable-pose refinement"
            self.add_to_ablation_study_pose_list(refinement_experiment_name, T_stable_in_cam[:3], np.array(anno['pose']))
        
        # 3. GT keypoints
        kpt_2d_gt = np.concatenate([anno['fps_2d'], [anno['center_2d']]], axis=0)

        # Calculate the euclidean distance between any of two kpts (pixels)
        distances = distance.cdist(kpt_2d_gt, kpt_2d_gt, 'euclidean')
        max_distance = np.max(distances)
        np.fill_diagonal(distances, np.inf)  # avoid comparing with itself
        min_distance = np.min(distances)
        self.max_euclidean_dis_bewteen_kpts = max(self.max_euclidean_dis_bewteen_kpts, max_distance)
        self.min_euclidean_dis_bewteen_kpts = min(self.min_euclidean_dis_bewteen_kpts, min_distance)

        pose_pred_from_kpts_gt = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_gt, K)
        experiment_name3 = "GT keypoints"
        self.add_to_ablation_study_pose_list(experiment_name3, pose_pred_from_kpts_gt, pose_gt)
        # self.average_error_kpt_gt_to_pnp(pose_pred_from_kpts_gt, pose_gt)

        # Helper function to apply random + or - noise
        def apply_random_noise(value, noise_amount):
            return value + random.choice([-noise_amount, noise_amount])
        
        # 4. 1-pixel-1-keypoint noise
        kpt_2d_1pixel_noise_1kpt = kpt_2d_gt.copy()
        random_kpt_idx = random.randint(0, 7)
        random_x_or_y = random.randint(0, 1)
        kpt_2d_1pixel_noise_1kpt[random_kpt_idx][random_x_or_y] = apply_random_noise(kpt_2d_1pixel_noise_1kpt[random_kpt_idx][random_x_or_y], 1)
        pose_pred_1pixel_noise_1kpt = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_1pixel_noise_1kpt, K)
        experiment_name4 = "adding ±1 pixel noise to x(or y) coordinate of a random keypoint"
        self.add_to_ablation_study_pose_list(experiment_name4, pose_pred_1pixel_noise_1kpt, pose_gt)

        # 5. 2-pixel-1-keypoint noise
        kpt_2d_2pixel_noise_1kpt = kpt_2d_gt.copy()
        random_kpt_idx = random.randint(0, 7)
        random_x_or_y = random.randint(0, 1)
        kpt_2d_2pixel_noise_1kpt[random_kpt_idx][random_x_or_y] = apply_random_noise(kpt_2d_2pixel_noise_1kpt[random_kpt_idx][random_x_or_y], 2)
        pose_pred_2pixel_noise_1kpt = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_2pixel_noise_1kpt, K)
        experiment_name5 = "adding ±2 pixel noise to x(or y) coordinate of a random keypoint"
        self.add_to_ablation_study_pose_list(experiment_name5, pose_pred_2pixel_noise_1kpt, pose_gt)

        # 6. 1-pixel-2-keypoints noise, same noise pattern for all kpts
        kpt_2d_1pixel_noise_2kpts = kpt_2d_gt.copy()
        random_kpt_idx_1 = random.randint(0, 7)
        random_x_or_y_1 = random.randint(0, 1)
        random_kpt_idx_2 = random.randint(0, 7)
        random_x_or_y_2 = random.randint(0, 1)
        kpt_2d_1pixel_noise_2kpts[random_kpt_idx_1][random_x_or_y_1] = apply_random_noise(kpt_2d_1pixel_noise_2kpts[random_kpt_idx_1][random_x_or_y_1], 1)
        kpt_2d_1pixel_noise_2kpts[random_kpt_idx_2][random_x_or_y_2] = apply_random_noise(kpt_2d_1pixel_noise_2kpts[random_kpt_idx_2][random_x_or_y_2], 1)
        pose_pred_1pixel_noise_2kpts = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_1pixel_noise_2kpts, K)
        experiment_name6 = "adding ±1 pixel noise to x(or y) coordinates of two random keypoints, same noise pattern for all kpts"
        self.add_to_ablation_study_pose_list(experiment_name6, pose_pred_1pixel_noise_2kpts, pose_gt)

        # 7. 2-pixel-2-keypoints noise, same noise pattern for all kpts
        kpt_2d_2pixel_noise_2kpts = kpt_2d_gt.copy()
        random_kpt_idx_1 = random.randint(0, 7)
        random_x_or_y_1 = random.randint(0, 1)
        random_kpt_idx_2 = random.randint(0, 7)
        random_x_or_y_2 = random.randint(0, 1)
        kpt_2d_2pixel_noise_2kpts[random_kpt_idx_1][random_x_or_y_1] = apply_random_noise(kpt_2d_2pixel_noise_2kpts[random_kpt_idx_1][random_x_or_y_1], 2)
        kpt_2d_2pixel_noise_2kpts[random_kpt_idx_2][random_x_or_y_2] = apply_random_noise(kpt_2d_2pixel_noise_2kpts[random_kpt_idx_2][random_x_or_y_2], 2)
        pose_pred_2pixel_noise_2kpts = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_2pixel_noise_2kpts, K)
        experiment_name7 = "adding ±2 pixel noise to x(or y) coordinates of two random keypoints, same noise pattern for all kpts"
        self.add_to_ablation_study_pose_list(experiment_name7, pose_pred_2pixel_noise_2kpts, pose_gt)

        # 8. 1-pixel-all-keypoints noise, same noise pattern for all kpts
        kpt_2d_1pixel_noise_all_kpts = kpt_2d_gt.copy()
        kpt_2d_1pixel_noise_all_kpts = kpt_2d_1pixel_noise_all_kpts + random.choice([-1, 1])
        pose_pred_1pixel_noise_all_kpts = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_1pixel_noise_all_kpts, K)
        experiment_name8 = "adding ±1 pixel noise to x(or y) coordinates of all keypoints, same noise pattern for all kpts"
        self.add_to_ablation_study_pose_list(experiment_name8, pose_pred_1pixel_noise_all_kpts, pose_gt)

        # 9. 2-pixel-all-keypoints noise, same noise pattern for all kpts
        kpt_2d_2pixel_noise_all_kpts = kpt_2d_gt.copy()
        kpt_2d_2pixel_noise_all_kpts = kpt_2d_2pixel_noise_all_kpts + random.choice([-2, 2])
        pose_pred_2pixel_noise_all_kpts = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_2pixel_noise_all_kpts, K)
        experiment_name9 = "adding ±2 pixel noise to x(or y) coordinates of all keypoints, same noise pattern for all kpts"
        self.add_to_ablation_study_pose_list(experiment_name9, pose_pred_2pixel_noise_all_kpts, pose_gt)

        # 10. 1-pixel-all-keypoints noise, different noise patterns for different kpts
        kpt_2d_1pixel_diff_noise_all_kpts = kpt_2d_gt.copy()
        for i in range(len(kpt_2d_1pixel_noise_all_kpts)):
            random_x_or_y = random.randint(0, 1)
            kpt_2d_1pixel_diff_noise_all_kpts[i][random_x_or_y] = apply_random_noise(kpt_2d_1pixel_diff_noise_all_kpts[i][random_x_or_y], 1)
        pose_pred_1pixel_diff_noise_all_kpts = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_1pixel_diff_noise_all_kpts, K)
        experiment_name10 = "adding ±1 pixel noise to x(or y) coordinates of all keypoints, different noise pattern for different kpts"
        self.add_to_ablation_study_pose_list(experiment_name10, pose_pred_1pixel_diff_noise_all_kpts, pose_gt)

        # 11. 2-pixel-all-keypoints noise, different noise patterns for different kpts
        kpt_2d_2pixel_diff_noise_all_kpts = kpt_2d_gt.copy()
        for i in range(len(kpt_2d_1pixel_noise_all_kpts)):
            random_x_or_y = random.randint(0, 1)
            kpt_2d_2pixel_diff_noise_all_kpts[i][random_x_or_y] = apply_random_noise(kpt_2d_2pixel_diff_noise_all_kpts[i][random_x_or_y], 2)
        pose_pred_2pixel_diff_noise_all_kpts = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_2pixel_diff_noise_all_kpts, K)
        experiment_name11 = "adding ±2 pixel noise to x(or y) coordinates of all keypoints, different noise pattern for different kpts"
        self.add_to_ablation_study_pose_list(experiment_name11, pose_pred_2pixel_diff_noise_all_kpts, pose_gt)

        # 12. 3-pixel-all-keypoints noise, different noise patterns for different kpts
        kpt_2d_3pixel_diff_noise_all_kpts = kpt_2d_gt.copy()
        for i in range(len(kpt_2d_1pixel_noise_all_kpts)):
            random_x_or_y = random.randint(0, 1)
            kpt_2d_3pixel_diff_noise_all_kpts[i][random_x_or_y] = apply_random_noise(kpt_2d_3pixel_diff_noise_all_kpts[i][random_x_or_y], 3)
        pose_pred_3pixel_diff_noise_all_kpts = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_3pixel_diff_noise_all_kpts, K)
        experiment_name12 = "adding ±3 pixel noise to x(or y) coordinates of all keypoints, different noise pattern for different kpts"
        self.add_to_ablation_study_pose_list(experiment_name12, pose_pred_3pixel_diff_noise_all_kpts, pose_gt)

        # 13. 4-pixel-all-keypoints noise, different noise patterns for different kpts
        kpt_2d_4pixel_diff_noise_all_kpts = kpt_2d_gt.copy()
        for i in range(len(kpt_2d_1pixel_noise_all_kpts)):
            random_x_or_y = random.randint(0, 1)
            kpt_2d_4pixel_diff_noise_all_kpts[i][random_x_or_y] = apply_random_noise(kpt_2d_4pixel_diff_noise_all_kpts[i][random_x_or_y], 4)
        pose_pred_4pixel_diff_noise_all_kpts = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_4pixel_diff_noise_all_kpts, K)
        experiment_name13 = "adding ±4 pixel noise to x(or y) coordinates of all keypoints, different noise pattern for different kpts"
        self.add_to_ablation_study_pose_list(experiment_name13, pose_pred_4pixel_diff_noise_all_kpts, pose_gt)

        # 14. 5-pixel-all-keypoints noise, different noise patterns for different kpts
        kpt_2d_5pixel_diff_noise_all_kpts = kpt_2d_gt.copy()
        for i in range(len(kpt_2d_1pixel_noise_all_kpts)):
            random_x_or_y = random.randint(0, 1)
            kpt_2d_5pixel_diff_noise_all_kpts[i][random_x_or_y] = apply_random_noise(kpt_2d_5pixel_diff_noise_all_kpts[i][random_x_or_y], 5)
        pose_pred_5pixel_diff_noise_all_kpts = pvnet_pose_utils.pnp(kpt_3d, kpt_2d_5pixel_diff_noise_all_kpts, K)
        experiment_name14 = "adding ±5 pixel noise to x(or y) coordinates of all keypoints, different noise pattern for different kpts"
        self.add_to_ablation_study_pose_list(experiment_name14, pose_pred_5pixel_diff_noise_all_kpts, pose_gt)

    def summarize_abalation_study_result(self):
        for experiment_name, pose_list in self.ablation_study_pose_list.items():
            trans_err_all = []
            angular_err_all = []
            
            for pose_gt, pose_pre in pose_list:
                t_pre = pose_pre[:, 3]
                t_gt = pose_gt[:, 3]
                translation_error = np.abs(t_pre - t_gt)
                
                R_pre_in_world = pose_pre[:, :3]
                R_gt_in_world = pose_gt[:, :3]
                R_pre_2_gt = np.dot(R_pre_in_world, R_gt_in_world.T)
                trace = np.trace(R_pre_2_gt)
                trace = min(3, max(-1, trace))
                angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
                if(angular_distance > 10): #if theta>10 deg, consider it as outlier
                    continue
                trans_err_all.append(translation_error)
                angular_err_all.append(angular_distance)
            
            trans_err_all = np.array(trans_err_all) * 1000 # m to mm
            angular_err_all = np.array(angular_err_all)
            
            trans_error_mean = np.mean(trans_err_all, axis=0)
            trans_error_std = np.std(trans_err_all, axis=0)
            
            angular_error_mean = np.mean(angular_err_all)
            angular_error_std = np.std(angular_err_all)
            
            print("========================================================")
            print("Below is the result of " + experiment_name + ": ")
            print('Translation Error (X-axis): {:.2f} mm, std {:.2f}'.format(trans_error_mean[0], trans_error_std[0]))
            print('Translation Error (Y-axis): {:.2f} mm, std {:.2f}'.format(trans_error_mean[1], trans_error_std[1]))
            print('Translation Error (Z-axis): {:.2f} mm, std {:.2f}'.format(trans_error_mean[2], trans_error_std[2]))
            print('Angular Error (rotation)  : {:.2f} deg, std {:.2f}'.format(angular_error_mean, angular_error_std))

    def summarize(self):
        print("="*100)
        print("Valid data cnt (after eliminating the outliers): {}".format(self.valid_data_cnt))
        print("Statistic of dataset scene:")
        mean_object_2_camera_dis = np.mean([T[2, 3] for T in self.T_gt])
        print('Average distance between object and camera: {:.2f} m'.format(mean_object_2_camera_dis))
        print('Max Eulidean distance between two keypoints: {:.2f} pixels'.format(self.max_euclidean_dis_bewteen_kpts))
        print('Min Eulidean distance between two keypoints: {:.2f} pixels'.format(self.min_euclidean_dis_bewteen_kpts))

        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)

        # 1. Raw output from PvNet
        trans_list = np.array(self.trans_err) * 1000 # m to mm
        trans_err = np.mean(trans_list, axis=0)
        trans_std = np.std(trans_list, axis=0)
        # angular_quat = np.mean(self.angular_quaternion_err)
        # angular_quat_std = np.std(self.angular_quaternion_err)
        angular_rotation = np.mean(self.angular_rotation_err)
        angular_rotation_std = np.std(self.angular_rotation_err)

        kpt_prediction_err = np.mean(self.avg_2d_kpts_error)
        kpt_prediction_std = np.std(self.avg_2d_kpts_error)

        print("========================================================")
        print("Below is the result from raw pvnet: ")
        print('Keypoint Prediction Error  : {:.2f} pix, std {:.2f}'.format(kpt_prediction_err, kpt_prediction_std))
        print('2d projections metric: {:.3f}'.format(proj2d))
        print('ADD metric: {:.3f}'.format(add))
        print('5 cm 5 degree metric: {:.3f}'.format(cmd5))
        print('mask ap90: {:.3f}'.format(ap))
        print('Translation Error (X-axis): {:.2f} mm, std {:.2f}'.format(trans_err[0], trans_std[0]))
        print('Translation Error (Y-axis): {:.2f} mm, std {:.2f}'.format(trans_err[1], trans_std[1]))
        print('Translation Error (Z-axis): {:.2f} mm, std {:.2f}'.format(trans_err[2], trans_std[2]))
        print('Angular Error (rotation)  : {:.2f} deg, std {:.2f}'.format(angular_rotation, angular_rotation_std))

        # 2. Raw output from PvNet with orientation refinement
        # TODO(yangfei): add stable pose logic here
        print("========================================================")
        print("Below is the result from raw pvnet with orientation refinement: ")

        # 3. Process and Report ablation study result
        self.summarize_abalation_study_result()

        print('='*100)
        # euler_err = np.mean(self.euler_err, axis=0)
        # print('Euler Angle Error (X-axis): {:.1f} deg'.format(euler_err[0]))
        # print('Euler Angle Error (Y-axis): {:.1f} deg'.format(euler_err[1]))
        # print('Euler Angle Error (Z-axis): {:.1f} deg'.format(euler_err[2]))

        # np.save(f"data/evaluation/trans_err.npy", trans_list)
        # np.save(f"data/evaluation/ang_R_err.npy", self.angular_rotation_err)
        # np.save(f"data/evaluation/ang_Q_err.npy", self.angular_quaternion_err)
        np.save(f"data/evaluation/T_gt.npy", self.T_gt)
        np.save(f"data/evaluation/T_pre.npy", self.T_pre)

        if self.icp_render is not None:
            print('ADD metric after icp: {:.3f}'.format(np.mean(self.icp_add)))
        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_add = []
        self.avg_2d_kpts_error = []
        return {'proj2d': proj2d, 'add': add, 'cmd5': cmd5, 'ap': ap, 'x_err_mm': trans_err[0], 'y_err_mm': trans_err[1], 'z_err_mm': trans_err[2], 'angular_err': angular_rotation, 'kpt_error': kpt_prediction_err}
