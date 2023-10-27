"""
This script uses Mask R-CNN and Pvnet to perform pose estimation
Author: Yangfei Dai and Hameed Abdul
"""
from mrcnn.utils.maskrcnnWrapper import MaskRCNNWrapper
import gin
import argparse
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

from yacs.config import CfgNode as CN
from lib.config import args, cfgs
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from lib.datasets.transforms import make_transforms
from lib.datasets import make_data_loader
from lib.utils.pvnet import pvnet_pose_utils


def predict_to_pose(pvnet_output, cfg, K_cam, input_img, is_vis: bool=False):
    kpt_3d = np.concatenate([cfg.fps_3d, [cfg.center_3d]], axis=0)
    kpt_2d = pvnet_output['kpt_2d'][0].detach().cpu().numpy()
    pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K_cam)

    print("Camera Intrinsics:")
    print(K_cam)
    print("Predicted Pose wrt camera:")
    print(pose_pred)

    if is_vis:
        visualize_pose(input_img, cfg, pvnet_output, K_cam, pose_pred)
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

def run_inference(pvnet, cfg, image, K_cam):
    pvnet.eval()

    transform = make_transforms(cfg, is_train=False)
    processed_image, _, _ = transform(image)
    processed_image = np.array(processed_image).astype(np.float32)

    # Convert the preprocessed image to a tensor and move it to GPU
    input_tensor = torch.from_numpy(processed_image).unsqueeze(0).cuda().float()

    with torch.no_grad():
        pvnet_output = pvnet(input_tensor)
    return predict_to_pose(pvnet_output, cfg, K_cam, image, is_vis=True)


# Configs/Models in the order 0: mainshell, 1: topshell, 2: insert_mold 
def make_and_load_pvnet(cfg):
    net = make_network(cfg).cuda()
    load_network(net, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    return net


def call_pvnet(data):
    K_cam = np.array([[10704.062350, 0, data['uv'][0]],
                [0, 10727.438047, data['uv'][1]],
                [0, 0, 1]])

    cat_idx = data['class']
    cur_pvnet = pvnets[cat_idx]
    cur_cfg = cfgs[cat_idx]
    cur_roi = data['image_128x128']
    return run_inference(cur_pvnet, cur_cfg, cur_roi, K_cam)


@gin.configurable
class CobotPoseEstNode(object):
    def __init__(self,
                 extrinsic_path: str=gin.REQUIRED):
        self.__dict__.update(locals())
        self._static_br = tf2_ros.StaticTransformBroadcaster()
        self._reset_robot_pub = rospy.Publisher("/reset_robot", Bool, queue_size=1)

        self.T_camera_in_base  = np.linalg.inv(np.load(extrinsic_path))
        print(self.T_camera_in_base)
        rospy.sleep(8)
        self.flagged = False

        self.tf_dict = {}

    def _publish_tf(self, pose_pred):
        """
        Publish transforms
        """

        for idx, pose in enumerate(pose_pred):
            n = np.linalg.norm(r)
            r = r / np.asarray(n)

            cls_name = self._cls_names[cls].replace(" ","_").lower()

            x1, y1, x2, y2 = r
            c = [n * (x2 + x1)*.5, n*(y2 + y1)*.5]
            center = [int(c[0]), int(c[1])]
            

            tf_name = f"predicted_part_pose/{idx}/{cls_name}"

            z_board2cam = self.T_camera_in_base[2, 3] - 0.02
            uv                   = np.array([*c, 1])
            t                    = np.linalg.inv(self.K) @ uv
            t                    *= z_board2cam

            P_part_in_cam       = np.array([*t, 1])
            P_part_in_base      = self.T_camera_in_base @ P_part_in_cam

            T_part_in_cam       = make_T(np.eye(3), *t)
            T_part_in_base      = self.T_camera_in_base @ T_part_in_cam
            trans               = T_part_in_base[:3, 3]

            q                    = mat2quat(T_part_in_base[:3, :3])
            q_ros                = quat_convention(q)


            # Publish transform
            publish_tf2(quat2mat(q), trans, 'world', tf_name)


            self.tf_dict[tf_name] = (quat2mat(q), trans)

            
            print(f"T_part_in_base\n{'='*50}\n{trans}")

        
        self.flagged = True


    def __call__(self, pvnet_predictions):
        """
        """
        while not rospy.is_shutdown():
            if not hasattr(self, '_im'):
                continue

            self._publish_tf(pvnet_predictions)
            cv2.imwrite(f'ros_img.png', out_img[:,:,::-1])
            cv2.imwrite(f'ros_input.png', self._im[:,:,::-1])

                
            rospy.sleep(0.5)
            reset_robot_flag = Bool()
            reset_robot_flag.data = True
            d2._reset_robot_pub.publish(reset_robot_flag)


if __name__ == '__main__':

    # Load all need models and configs
    gin.parse_config_file('./mrcnn/maskrcnn_config.gin')
    mrcnn = MaskRCNNWrapper()

    pvnets = tuple([make_and_load_pvnet(c) for c in cfgs])

    # TODO: replace. Just for demo purposes
    img = cv2.cvtColor(cv2.imread('./mrcnn/inference_exp/new_ws_oct23/Image__2023-09-26__18-10-34.png'), cv2.COLOR_BGR2RGB)

    data_for_pvnet, _, _ = mrcnn(img)

    poses = [call_pvnet(data) for data in data_for_pvnet]
    print(poses)
