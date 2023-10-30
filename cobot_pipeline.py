"""
This script uses Mask R-CNN and Pvnet to perform pose estimation
Author: Yangfei Dai and Hameed Abdul
"""
from typing import List, Dict

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
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat
from cv_bridge import CvBridge, CvBridgeError

from yacs.config import CfgNode as CN
from lib.config import args, cfgs
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from lib.datasets.transforms import make_transforms
from lib.datasets import make_data_loader
from lib.utils.pvnet import pvnet_pose_utils
from mrcnn.utils.maskrcnnWrapper import MaskRCNNWrapper


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
    ax.imshow(input_img)

    # Calculate center of bounding box
    center_x = np.mean(corner_2d_pred[:, 0])
    center_y = np.mean(corner_2d_pred[:, 1])
    shift_x = center_x - corner_2d_pred[6, 0]
    shift_y = center_y - corner_2d_pred[6, 1]
    # Plot X-axis
    ax.plot([center_x , corner_2d_pred[2, 0]+shift_x], [center_y, corner_2d_pred[2, 1]+shift_y], color='r', linewidth=1)
    # Plot Y-axis
    ax.plot([center_x, corner_2d_pred[4, 0]+shift_x], [center_y, corner_2d_pred[4, 1]+shift_y], color='g', linewidth=1)
    # Plot Z-axis
    ax.plot([center_x, corner_2d_pred[7, 0]+shift_x], [center_y, corner_2d_pred[7, 1]+shift_y], color='b', linewidth=1)
    # Add patches for corner_2d_gt and corner_2d_pred
    ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
    ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
    plt.axis('off')
    plt.title('Pose Prediction')
    # plt.savefig("/pvnet/data/evaluation/topshell.png")

    ###########################
    # vertex: currently makes no sense
    ###########################
    # plt.figure(1)
    # from torchvision.utils import make_grid
    # import torchvision
    # Grid = make_grid(output['vertex'].permute(1,0,2,3), nrow=9, padding=25)
    # vector_map = torchvision.transforms.ToPILImage()(Grid.cpu())
    # vector_map.show()
    # plt.imshow(vector_map)

    ###########################
    # segmentaion map, note:
    # mask = torch.argmax(output['seg'], 1)
    ###########################
    plt.figure(2)
    plt.subplot(121)
    plt.imshow(segmentation[0])
    plt.axis('off')
    plt.title('Segmentaion 1')

    plt.subplot(122)
    plt.imshow(segmentation[1])
    plt.axis('off')
    plt.title('Segmentaion 2')

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
    K_cam = np.array([[10704.062350, 0, data['uv'][0]],
                [0, 10727.438047, data['uv'][1]],
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
                 T_camera_in_base: str=gin.REQUIRED):
        self.__dict__.update(locals())
        self._static_br = tf2_ros.StaticTransformBroadcaster()
        self._reset_robot_pub = rospy.Publisher("/reset_robot", Bool, queue_size=1)

        self._cls_names = ("Main shell", "Top shell", "Insert Mold")
        self.T_camera_in_base  = np.load(T_camera_in_base)
        print(f"T_camera_in_base\n {self.T_camera_in_base}\n{'='*50}")
        rospy.sleep(8)
        self.flagged = False
        rospy.init_node(node_name)

        self.tf_dict = {}

    def _publish_tf(self, pvnet_outputs, pvnet_inputs):
        """
        Publish transforms
        """
        for idx, (T_part_in_cam, input_data) in enumerate(zip(pvnet_outputs, pvnet_inputs)):

            cls = input_data["class"]
            cls_name = self._cls_names[cls].replace(" ","_").lower()


            tf_name = f"predicted_part_pose/{idx}/{cls_name}"
            # TODO (ham): measure offset and add here. You shold project to camera frame, replace the z value of each part and then project to back to the robot frame
            T_part_in_base  = self.T_camera_in_base @ T_part_in_cam

            # TODO: replace with the hardcoded z value
            T_part_in_base[2, 3] = 0.1

            R        = T_part_in_base[:3, :3]
            t        = T_part_in_base[:3, 3]
            
            # Publish transform
            publish_tf2(R, t, 'world', tf_name)
            self.tf_dict[tf_name] = (R, t)           
            print(f"T_part_in_base[{idx}]\n{T_part_in_base}\n{'='*50}")

        
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
