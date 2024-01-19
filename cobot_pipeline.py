"""
Purpose: This is a ROS script that contains the pose estimation pipeline. Pose estimation is a two stage process: 1) Mask R-CNN + 2) PVNet
Author: Hameed Abdul (hameeda2@illinois.edu) and Yangfei Dai (yangfei4@illinois.edu)
"""
from typing import List, Dict, Optional

import gin
import argparse
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import message_filters
import rospy
import tf
import tf2_ros
import geometry_msgs.msg
from PIL import Image
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Bool
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import mat2quat, quat2mat
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation

from yacs.config import CfgNode as CN
from lib.config import args, cfgs
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from lib.datasets.transforms import make_transforms
from lib.datasets import make_data_loader
from lib.utils.pvnet import pvnet_pose_utils
import mrcnn.utils.visualization as vis
from mrcnn.utils.maskrcnnWrapper import MaskRCNNWrapper
from mrcnn.stable_poses_est import find_closest_stable_pose, construct_T_stable_from_T_est
           

# Camera Intrinsics
# Instrinsics April 2023 - 25mm lens
# See https://github.com/cobot-factory/ur5e_collab_ws/blob/main/src/cobot_resources/camera/basler_25mm_intrinsics_April.28.23.yaml
camera = {
    "fx": 10704.062350, 
    "fy": 10727.438047, 
    "ox": 2694.112343, 
    "oy": 1669.169773, 
    "w": 5472, 
    "h": 3648 
}

K = np.stack([[camera['fx'], 0, camera['ox']],
              [0, camera['fy'], camera['oy']],
              [0,            0,           1]])

def sample_points(geo_path, num_points=10000):
    import open3d as o3d 
    # from open3d.geometry import TriangleMesh as tri
    # from open3d.io import read_triangle_mesh
    ply = o3d.io.read_triangle_mesh(geo_path)
    pts = np.array(o3d.geometry.TriangleMesh.sample_points_uniformly(ply, num_points).points)
    return np.block([pts, np.ones((num_points, 1))]).T

def uv_2_xyz(uv, t_part_in_base, T_camera_in_base):
    _, _, z_part_in_base = t_part_in_base
    z_scale = T_camera_in_base[2, 3] - z_part_in_base 
    XYZ_cam = z_scale * np.linalg.inv(K) @ np.array([*uv, 1]).T 
    XYZ_base = T_camera_in_base @ np.array([*XYZ_cam, 1]) 
    return XYZ_base[:3]

def draw_cad_model(T_part_in_base, cls, img, T_base_in_cam, crop_dim=64):

    paths_to_geo = ("./data/FIT/mainshell_test/model.ply",
                    "./data/FIT/topshell_test/model.ply",
                    "./data/FIT/insert_mold_test/model.ply") 
    geo_path = paths_to_geo[cls]
    P_part = sample_points(geo_path)

    P = K @ np.eye(3, 4) @ T_base_in_cam
    cent_p = P @ T_part_in_base[:, 3]
    cent_pix = (cent_p / cent_p[2])[:2].astype(np.uint)
    
    
    im_pix = P @ T_part_in_base @ P_part
    im_pix /= im_pix[2]
    
    ind = im_pix[:2].astype(np.uint)
    
    img[ind[1, :], ind[0, :]] = 255
    
    crop_roi = vis.crop_roi(img, cent_pix.tolist(), crop_dim)
    return img, crop_roi

def predict_to_pose(pvnet_output, cfg, K_cam, input_img, is_vis: bool=False, is_pose_H: bool=True):
    kpt_3d = np.concatenate([cfg.fps_3d, [cfg.center_3d]], axis=0)
    kpt_2d = pvnet_output['kpt_2d'][0].detach().cpu().numpy()
    pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K_cam)

    if is_vis:
        visualize_pose(input_img, cfg, pvnet_output, K_cam, pose_pred)

    if is_pose_H:
        # return pose as 4x4 matrix
        return np.c_[pose_pred.T, np.array([0, 0, 0, 1])].T
    # return pose as 3x4 matrix
    return pose_pred

def draw_axis(img, R, t, K, scale=0.006, dist=None):
    """
    Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
    :param img - rgb numpy array
    :R - Rotation matrix, 3x3
    :t - 3d translation vector, in meters (dtype must be float)
    :K - intrinsic calibration matrix , 3x3
    :scale - factor to control the axis lengths
    :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
    """
    img = img.astype(np.float32)
    rotation_vec, _ = cv2.Rodrigues(R) #euler rotations
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)
    
    axis_points = axis_points.astype(int)
    corner = tuple(axis_points[3].ravel())
    img = cv2.line(img, corner, tuple(axis_points[0].ravel()), (255, 0, 0), 1)
    # img = cv2.putText(img, "X", tuple(axis_points[0].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 1)

    img = cv2.line(img, corner, tuple(axis_points[1].ravel()), (0, 255, 0), 1)
    # img = cv2.putText(img, "Y", tuple(axis_points[1].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 1)

    img = cv2.line(img, corner, tuple(axis_points[2].ravel()), (0, 0, 255), 1)

    img = img.astype(np.uint8)
    return img

def visualize_reprojection(input_roi, raw_pvnet_roi, stable_pose_roi, refined_position_roi):
    figsize=(14, 14)
    fig = plt.figure(figsize=figsize,tight_layout=True)
    plt.subplot(141)
    plt.imshow(input_roi)
    plt.axis('off')
    plt.title('Input Image')

    plt.subplot(142)
    plt.imshow(raw_pvnet_roi)
    plt.axis('off')
    plt.title('Reprojection of PVNet Predicted Pose')

    plt.subplot(143)
    plt.imshow(stable_pose_roi)
    plt.axis('off')
    plt.title('Reprojection of Stable Pose')
                            
    plt.subplot(144)
    plt.imshow(refined_position_roi)
    plt.axis('off')
    plt.title('Reprojection of Stable Pose & Refined Pos.')

    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    fig.set_tight_layout(True)
    plt.show()

def visualize_pose(input_img, cfg, pvnet_output, K_cam, pose_pred):
    corner_3d = cfg.corner_3d
    kpt_2d = pvnet_output['kpt_2d'][0].detach().cpu().numpy()
    segmentation = pvnet_output['seg'][0].detach().cpu().numpy()
    mask = pvnet_output['mask'][0].detach().cpu().numpy()
    corner_2d_pred = pvnet_pose_utils.project(corner_3d, K_cam, pose_pred)

    ###########################
    # overall result
    ###########################
    plt.figure(0)
    plt.subplot(221)
    plt.imshow(input_img)
    plt.axis('off')
    plt.title('Input Image')

    plt.subplot(222)
    plt.imshow(mask)
    plt.axis('off')
    plt.title('Predicted Mask')

    plt.subplot(223)
    plt.imshow(input_img)
    plt.scatter(kpt_2d[:8, 0], kpt_2d[:8, 1], color='red', s=10)
    plt.axis('off')
    plt.title('Key points detection')

    ax = plt.subplot(224)
    # ax.imshow(input_img)
    ax.imshow(draw_axis(input_img, pose_pred[:3, :3], pose_pred[:3, 3], K_cam))

    plt.axis('off')
    plt.title('Pose Prediction')
    # plt.savefig("/pvnet/data/evaluation/topshell.png")

    from scipy.spatial.transform import Rotation
    R = pose_pred[:3, :3]
    euler_angles = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    euler_angles_rounded = [int(angle) for angle in euler_angles]
    print("Euler angles for Estimated Pose in camera frame:", euler_angles_rounded)

    plt.show()

def run_inference(pvnet, cfg, image, K_cam, is_vis=True):
    pvnet.eval()

    transform = make_transforms(cfg, is_train=False)
    processed_image, _, _ = transform(image)
    processed_image = np.array(processed_image).astype(np.float32)

    # Convert the preprocessed image to a tensor and move it to GPU
    input_tensor = torch.from_numpy(processed_image).unsqueeze(0).cuda().float()

    with torch.no_grad():
        pvnet_output = pvnet(input_tensor)
    return predict_to_pose(pvnet_output, cfg, K_cam, image, is_vis=is_vis)


# Configs/Models in the order 0: mainshell, 1: topshell, 2: insert_mold 
def make_and_load_pvnet(cfg):
    net = make_network(cfg).cuda()
    load_network(net, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    return net


def call_pvnet(data, is_vis=True):
    cam_u = 2694.112343
    cam_v = 1669.169773

    W, H = int(5472), int(3648)
    crop_size = 128
    # shift the uv from original camera uv to cropped image center
    shifted_u = cam_u + (W//2 - data['uv'][0]) - (W//2 - crop_size//2)
    shifted_v = cam_v + (H//2 - data['uv'][1]) - (H//2 - crop_size//2)

    K_cam = np.array([[10704.062350, 0, shifted_u],
                [0, 10727.438047, shifted_v],
                [0, 0, 1]])

    cat_idx = data['class']
    cur_pvnet = pvnets[cat_idx]
    cur_cfg = cfgs[cat_idx]
    cur_roi = data['image_128x128']
    return run_inference(cur_pvnet, cur_cfg, cur_roi, K_cam, is_vis)


def publish_tf2(R: np.ndarray, t: np.array, parent_frame: str, child_frame: str):
    assert R.shape == (3, 3), f"Rot matrix shape is ({R.shape}) not (3,3)"

    br = tf2_ros.TransformBroadcaster()
    trans_info = geometry_msgs.msg.TransformStamped()
    trans_info.header.stamp    = rospy.Time.now()
    trans_info.header.frame_id = parent_frame
    trans_info.child_frame_id  =  child_frame

    x, y, z = t
    trans_info.transform.translation.x = x
    trans_info.transform.translation.y = y
    trans_info.transform.translation.z = z

    q = quat_convention(mat2quat(R), trans3d=False)
    trans_info.transform.rotation.x = q[0]
    trans_info.transform.rotation.y = q[1]
    trans_info.transform.rotation.z = q[2]
    trans_info.transform.rotation.w = q[3]
    br.sendTransform(trans_info)


def quat_convention(q: list, trans3d: bool=False)-> list:
    if trans3d:
        x, y, z, w = q
        return [w, x, y, z]
    # Currently mapping to w,x,y,z to x,y,z,w
    w, x, y, z  = q
    return [x, y, z, w]


@gin.configurable
class CobotPoseEstNode(object):
    def __init__(self,
                 node_name: str,
                 T_camera_in_base: str=gin.REQUIRED,
                 T_tagboard_in_camera: str=gin.REQUIRED,
                 _im: Optional[np.ndarray]=None,
                 ):
        self.__dict__.update(locals())
        self._static_br = tf2_ros.StaticTransformBroadcaster()
        self._reset_robot_pub = rospy.Publisher("/reset_robot", Bool, queue_size=1)

        self._cls_names = ("Main shell", "Top shell", "Insert Mold")
        self.T_camera_in_base  = np.load(T_camera_in_base)
        
        # Since the pre-computed T_tagboard_in_camera is in pointing down in z direction, 
        # which is unexpected, we need to convert it to the tagUp frame
        self.T_tagboard_in_camera = np.load(T_tagboard_in_camera)
        
        self.T_base_in_camera = np.linalg.inv(self.T_camera_in_base)
        self.T_camera_in_tagboard = np.linalg.inv(self.T_tagboard_in_camera)

        self.T_base_in_tagboard = self.T_camera_in_tagboard @ self.T_base_in_camera

        inch2m = 0.0254
        adpaterplate_thickness = 1 * inch2m
        tagboard_thickness = 0.125 * inch2m
        Z_OFFSET_TAG_IN_BASE = -0.875 * inch2m # 1" adapter plate - 0.125" tagboard thickness

        self.T_tagboard_in_base = np.eye(4)
        self.T_tagboard_in_base[2, 3] = Z_OFFSET_TAG_IN_BASE


        print(f"T_camera_in_base\n {self.T_camera_in_base}\n{'='*50}")
        rospy.sleep(8)
        self.flagged = False
        rospy.init_node(node_name)

        self.tf_dict = {'camera_in_base': (self.T_camera_in_base[:3, :3], self.T_camera_in_base[:3, 3]),
                        'tagboard_in_base': (self.T_tagboard_in_base[:3, :3], self.T_tagboard_in_base[:3, 3])}

    def _publish_tf(self, pvnet_outputs, pvnet_inputs):
        """
        Publish transforms
        """
        debug = False
        if debug:
            for idx, (T_part_in_cam, input_data) in enumerate(zip(pvnet_outputs, pvnet_inputs)):

                cls = input_data["class"]
                cls_name = self._cls_names[cls].replace(" ","_").lower()


                tf_name = f"stable_pose_in_base/{idx}/{cls_name}"
                uv = input_data["uv"]

                K =  np.array([[10704.062350, 0, 2694.112343 ],
                               [0, 10727.438047, 1669.169773],
                               [0, 0, 1]])

                # convert to normalized image coordinates
                u_norm = (uv[0] - K[0,2])/K[0,0]
                v_norm = (uv[1] - K[1,2])/K[1,1]

                # create 3D vector in camera frame
                xyz_cam = np.array([u_norm, v_norm, 1])


                # transform to base frame
                XYZ_world = self.T_camera_in_base @ np.array([*xyz_cam, 1])

                R = np.eye(3)
                print(f"XYZ_world: {XYZ_world}")
                t = XYZ_world[:3]
                t[2] = 0.1

                print(f"uv: {uv}  |Projected uv: {t}")

                # Publish transform
                publish_tf2(R, t, 'world', tf_name)
                self.tf_dict[tf_name] = (R, t)           
                print(f"uv: {uv}  |Projected uv: {t}")
        else:

            vis_queue = []
            for idx, (T_part_in_cam, input_data) in enumerate(zip(pvnet_outputs, pvnet_inputs)):

                cls = input_data["class"]
                cls_name = self._cls_names[cls].replace(" ","_").lower()


                tf_name = f"stable_pose_in_base/{idx}/{cls_name}"
                # TODO (ham): measure offset and add here. You shold project to camera frame, replace the z value of each part and then project to back to the robot frame
                T_part_in_base = self.T_camera_in_base @ T_part_in_cam
                T_part_in_tagboard = np.linalg.inv(self.T_tagboard_in_base)  @ T_part_in_base

                T_stable_part_in_tagboard = construct_T_stable_from_T_est(T_part_in_tagboard, cls)
                T_stable_part_in_base = self.T_tagboard_in_base @ T_stable_part_in_tagboard
                T_old_stable_part_pose = T_stable_part_in_base.copy()

                R        = T_stable_part_in_base[:3, :3]
                # Use Mask R-CNN's xy predictions with a known Z value 
                t        = uv_2_xyz(input_data["uv"], T_stable_part_in_base[:3, 3], self.T_camera_in_base)
                t  = np.array([*t[:2], T_stable_part_in_base[2, 3]])

                T_stable_part_in_base[:3, 3] = t

                input_roi = input_data["image_128x128"]
                _, raw_roi = draw_cad_model(T_part_in_base, cls, self._im.copy(), self.T_base_in_camera)
                _, stable_roi = draw_cad_model(T_old_stable_part_pose, cls, self._im.copy(), self.T_base_in_camera)
                _, corrected_roi = draw_cad_model(T_stable_part_in_base, cls, self._im.copy(), self.T_base_in_camera)

                vis_queue.append((input_roi, raw_roi, stable_roi, corrected_roi))
                
                # Publish transform
                publish_tf2(R, t, 'world', tf_name)
                self.tf_dict[tf_name] = (R, t)
                print(f"{'='*50}\n{tf_name}\n{'='*50}")
                print(f"T_part_in_base[{idx}]\n{T_part_in_base}\n{'='*50}")
                print(f"BEFORE Corection T_stable_part_in_base[{idx}]\n{T_old_stable_part_pose}\n{'='*50}")
                print(f"AFTER Correction T_stable_part_in_base[{idx}]\n{T_stable_part_in_base}\n{'='*50}")

            [visualize_reprojection(*vis) for vis in vis_queue]


        
        self.flagged = True


    def __call__(self, pvnet_predictions: List, pvnet_inputs: Dict):
        """
        """
        while not rospy.is_shutdown():
            if not hasattr(self, '_im'):
                print("No image")
                continue
            
            if self.flagged:
                for tf_name, (R,t) in self.tf_dict.items():
                    publish_tf2(R, t, 'world', tf_name)
            else:
                self._publish_tf(pvnet_predictions, pvnet_inputs)
                cv2.imwrite(f'ros_input.png', self._im[:,:,::-1])
                np.set_printoptions(precision=4)
                for tf_name, (R,t) in self.tf_dict.items():
                    euler = np.array([ax * 180 / np.pi for ax in mat2euler(R)])
                    quat = mat2quat(R)
                    print(f"{tf_name:<50}: XYZ pos {t} | Euler {euler}")
                    
            rospy.sleep(0.5)
            reset_robot_flag = Bool()
            reset_robot_flag.data = True
            self._reset_robot_pub.publish(reset_robot_flag)


if __name__ == '__main__':

    # Load all need models and configs
    gin.parse_config_file('./mrcnn/maskrcnn_config.gin')
    mrcnn = MaskRCNNWrapper()


    # TODO: replace. Just for demo purposes
    # img = cv2.cvtColor(cv2.imread('./mrcnn/inference_exp/new_ws_oct23/Image__2023-09-26__18-10-34.png'), cv2.COLOR_BGR2RGB)

    # init ros node
    pose_node = CobotPoseEstNode()
    
    # Subscribe to camera topic and wait for image
    print('***Mask R-CNN ready, waiting for camera images***')
    cam_topic = '/pylon_camera_node' 
    rgb_sub = message_filters.Subscriber(f'{cam_topic}/image_raw', Image, queue_size=1)
    msg = rospy.wait_for_message(f'{cam_topic}/image_raw', Image)
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub], queue_size=1, slop=0.1)
    cv_bridge = CvBridge()
    img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    data_for_pvnet, _, _ = mrcnn(img)

    pvnets = tuple([make_and_load_pvnet(c) for c in cfgs])
    poses = [call_pvnet(data, is_vis=True) for data in data_for_pvnet]

    # TODO: replace. Just for demo purposes
    pose_node._im = img
    pose_node(poses, data_for_pvnet)
