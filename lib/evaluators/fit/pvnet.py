from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_pose_utils, pvnet_data_utils
import os
from lib.utils.linemod import linemod_config
import torch
if cfg.test.icp:
    from lib.utils import icp_utils
from PIL import Image
from lib.utils.img_utils import read_depth
from scipy import spatial
from scipy.spatial.transform import Rotation

class Evaluator:

    def __init__(self, result_dir):
        self.result_dir = result_dir
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

        data_root = args['data_root']
        cls = cfg.cls_type
        model_path = os.path.join('data', cls+'_train', 'model.ply')
        self.model = pvnet_data_utils.get_ply_model(model_path)
        # self.diameter = np.loadtxt('data/custom/diameter.txt').item()

        self.proj2d = []
        self.add = []
        self.icp_add = []
        self.cmd5 = []
        self.mask_ap = []
        # self.euler_err = [] # degree
        self.angular_rotation_err = [] # degree
        self.angular_quaternion_err = [] # degree
        self.trans_err = [] # meter
        self.icp_render = icp_utils.SynRenderer(cfg.cls_type) if cfg.test.icp else None

    def average_error(self, pose_pred, pose_gt):
        # This line is to make sure the Z Value is positive(object points towards the camera)
        if(pose_pred[2,3]<0):
            pose_pred *= -1
        translation_error = np.abs(pose_pred[:, 3] - pose_gt[:, 3])

        # rotation_pred = Rotation.from_matrix(pose_pred[:, :3])
        # rotation_gt = Rotation.from_matrix(pose_gt[:, :3])
        # euler_error = rotation_pred.inv() * rotation_gt
        # euler_error = euler_error.as_euler('zyx', degrees=True)
        # euler_error = np.abs(euler_error)

        self.trans_err.append(translation_error)
        # self.euler_err.append(euler_error)

    def projection_2d(self, pose_pred, pose_targets, K, threshold=5):
        model_2d_pred = pvnet_pose_utils.project(self.model, K, pose_pred)
        model_2d_targets = pvnet_pose_utils.project(self.model, K, pose_targets)
        proj_mean_diff = np.mean(np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1))

        self.proj2d.append(proj_mean_diff < threshold)

    def add_metric(self, pose_pred, pose_targets, icp=False, syn=False, percentage=1):
        # diameter = self.diameter * percentage
        diameter = 10 / 1000
    
        # This line is to make sure the Z Value is positive(object points towards the camera)
        if(pose_pred[2,3]<0):
            pose_pred *= -1

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

    def quaternion_angular_err(self, pose_pred, pose_targets):
        # This line is to make sure the Z Value is positive(object points towards the camera)
        if(pose_pred[2,3]<0):
            pose_pred *= -1
        R1 = pose_pred[:, :3]
        R2 = pose_targets[:, :3]

        # Convert rotation matrices to quaternion representations
        Q1 = Rotation.from_matrix(R1).as_quat()
        Q2 = Rotation.from_matrix(R2).as_quat()

        # Compute the dot product between the quaternions
        # quat_diff = np.abs(np.multiply(quat1, quat2))
        # Compute the dot product between Q1 and Q2.conjugate
        Q2_conjugate = np.array([Q2[0], -Q2[1], -Q2[2], -Q2[3]])
        

        def dot_product(q1, q2):
            return q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]

        def quaternion_multiply(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2

            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
            z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

            return np.array([w, x, y, z])

        # Compute the angular difference (in radians)
        # Q_diff = quaternion_multiply(Q1, Q2_conjugate)
        # angular_diff = 2 * np.arccos(Q_diff[0])
    
        Q_diff = dot_product(Q1, Q2)
        angular_diff = 2 * np.arccos(Q_diff)

        # Convert angular difference to degrees
        angular_diff_deg = np.rad2deg(angular_diff)
        self.angular_quaternion_err.append(angular_diff_deg)

    def cm_degree_5_metric(self, pose_pred, pose_targets):
        # This line is to make sure the Z Value is positive(object points towards the camera)
        if(pose_pred[2,3]<0):
            pose_pred *= -1
        translation_distance = np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        trace = trace if trace >= -1 else -1
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        self.angular_rotation_err.append(angular_distance)
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

    def evaluate(self, output, batch):
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])
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
        self.average_error(pose_pred, pose_gt)
        self.quaternion_angular_err(pose_pred, pose_gt)

    def summarize(self):
        proj2d = np.mean(self.proj2d)
        add = np.mean(self.add)
        cmd5 = np.mean(self.cmd5)
        ap = np.mean(self.mask_ap)
        trans_err = np.mean(self.trans_err, axis=0) * 1000
        trans_std = np.std(self.trans_err, axis=0)

        angular_quat = np.mean(self.angular_quaternion_err)
        angular_quat_std = np.std(self.angular_quaternion_err)

        angular_rotation = np.mean(self.angular_rotation_err)
        angular_rotation_std = np.std(self.angular_rotation_err)

        print('2d projections metric: {:.3f}'.format(proj2d))
        print('ADD metric: {:.3f}'.format(add))
        print('5 cm 5 degree metric: {:.3f}'.format(cmd5))
        print('mask ap90: {:.3f}'.format(ap))

        print('Translation Error (X-axis): {:.2f} mm, std {:.3f}'.format(trans_err[0], trans_std[0]))
        print('Translation Error (Y-axis): {:.2f} mm, std {:.3f}'.format(trans_err[1], trans_std[1]))
        print('Translation Error (Z-axis): {:.2f} mm, std {:.3f}'.format(trans_err[2], trans_std[2]))

        print('Angular Error (quaternion): {:.2f} deg, std {:.3f}'.format(angular_quat, angular_quat_std))
        print('Angular Error (rotation)  : {:.2f} deg, std {:.3f}'.format(angular_rotation, angular_rotation_std))

        # euler_err = np.mean(self.euler_err, axis=0)
        # print('Euler Angle Error (X-axis): {:.1f} deg'.format(euler_err[0]))
        # print('Euler Angle Error (Y-axis): {:.1f} deg'.format(euler_err[1]))
        # print('Euler Angle Error (Z-axis): {:.1f} deg'.format(euler_err[2]))

        if self.icp_render is not None:
            print('ADD metric after icp: {:.3f}'.format(np.mean(self.icp_add)))
        self.proj2d = []
        self.add = []
        self.cmd5 = []
        self.mask_ap = []
        self.icp_add = []
        return {'proj2d': proj2d, 'add': add, 'cmd5': cmd5, 'ap': ap}
