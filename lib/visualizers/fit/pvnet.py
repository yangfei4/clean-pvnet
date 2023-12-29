from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg as default_cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
import cv2


mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self, cfg=None):
        self.cfg = cfg or default_cfg
        dataset_log = DatasetCatalog(cfg_new=self.cfg)
        args = dataset_log.get(self.cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])
        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        # shift predicted location to Ground True
        # pose_pred[:,3] = pose_gt[:,3]
        self.compute_dif(pose_pred, pose_gt)

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
        fig, ax = plt.subplots(1)

        # draw x y and z axes based on Rotation matrix
        # inp = draw_axis(img=inp.cpu().numpy(), R=pose_pred[:3, :3], t=pose_pred[:, 3], K=K)

        ax.imshow(inp)
        ax.axis("off")

        from scipy.spatial.transform import Rotation
        R = pose_pred[:3, :3]
        euler_angles = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        euler_angles_rounded = [int(angle) for angle in euler_angles]
        print("Euler angles for Estimated Pose:", euler_angles_rounded)

        ##### Crop around centers of corner_2d_gt #####
        # Calculate the center of corner_2d_gt
        center_x = np.mean(corner_2d_gt[:, 0])
        center_y = np.mean(corner_2d_gt[:, 1])
        # Define the crop size
        crop_size = 50  # Adjust the size as needed
        # Set the limits for the x-axis and y-axis
        ax.set_xlim(center_x - crop_size, center_x + crop_size)
        ax.set_ylim(center_y - crop_size, center_y + crop_size)
        ################################################


        # Calculate center of bounding box
        center_x = np.mean(corner_2d_pred[:, 0])
        center_y = np.mean(corner_2d_pred[:, 1])
        shift_x = center_x - corner_2d_pred[6, 0]
        shift_y = center_y - corner_2d_pred[6, 1]
        # Plot X-axis
        ax.plot([center_x , corner_2d_pred[2, 0]+shift_x], [center_y, corner_2d_pred[2, 1]+shift_y], color='r', linewidth=1)
        ax.plot([center_x , corner_2d_gt[2, 0]+shift_x], [center_y, corner_2d_gt[2, 1]+shift_y], color='r', linewidth=1)
        # Plot Y-axis
        ax.plot([center_x, corner_2d_pred[4, 0]+shift_x], [center_y, corner_2d_pred[4, 1]+shift_y], color='g', linewidth=1)
        ax.plot([center_x, corner_2d_gt[4, 0]+shift_x], [center_y, corner_2d_gt[4, 1]+shift_y], color='g', linewidth=1)
        # Plot Z-axis
        ax.plot([center_x, corner_2d_pred[7, 0]+shift_x], [center_y, corner_2d_pred[7, 1]+shift_y], color='b', linewidth=1)
        ax.plot([center_x, corner_2d_gt[7, 0]+shift_x], [center_y, corner_2d_gt[7, 1]+shift_y], color='b', linewidth=1)

        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        plt.show()
        return fig

    def compute_dif(self, pose_pred, pose_targets):
        translation_distance = (pose_pred[:, 3] - pose_targets[:, 3]) * 1000 # mm
        rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        trace = trace if trace >= -1 else -1
        angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
        print("predicted pose: {}".format(pose_pred))
        print("GT pose: {}".format(pose_targets))
        print("Rotation diff matrix {}".format(rotation_diff))
        print("Translation error: {} mm".format(translation_distance))
        print("Angular error: {} deg".format(angular_distance))

    def visualize_output(self, input_img, output, batch_example, K_cam=None):
        # output: 
        # 'seg'    : 1 x 2 x img_size x img_size
        # 'vertex' : 1 x 18 x img_size x img_size
        # 'mask'   : 1 x img_size x img_size
        # 'kpt_2d' : 1 x 9 x 2
        # 'var'    : 1 x 9 x 2 x 2 
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        segmentation = output['seg'][0].detach().cpu().numpy()
        mask = output['mask'][0].detach().cpu().numpy()
        # vertex = output['vertex'][0][0].detach().cpu().numpy()
        
        # K = np.array([[1.90856e+03, 0.00000e+00, 1.28000e+02/2],
        #               [0.00000e+00, 1.90906e+03, 1.28000e+02/2],
        #               [0.00000e+00, 0.00000e+00, 1.00000e+00]])
        # K = np.array([[21971.333024, 0, 1.28000e+02/2], 
        #               [0, 22025.144687, 1.28000e+02/2],
        #               [0, 0, 1]])
        
        if K_cam is None:
            K_cam = np.array([[10704.062350, 0, 2107+64], 
                      [0, 10727.438047, 1323+64],
                      [0, 0, 1]])

        img_id = int(batch_example['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        corner_3d = np.array(anno['corner_3d'])

        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K_cam)

        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K_cam, pose_pred)
        print("Camera Intrinsics:")
        print(K_cam)
        print("Predicted Pose wrt camera:")
        print(pose_pred)

        euler_angles = self.pose_matrix_to_euler(pose_pred)
        print("Euler angle:")
        print(euler_angles)

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


        ######################################################
        # logic to save a sub-figure as a numpy array
        ######################################################
        import io
        from PIL import Image
        # ax = plt.subplot(224)  # this is the subplot you want to save
        buf = io.BytesIO()  # create an in-memory binary stream
        ax.figure.savefig(buf, format='png', bbox_inches='tight')  # save figure to binary stream
        buf.seek(0)  # reset the position of the stream to the beginning
        im = Image.open(buf)  # use PIL to read in the image data
        output_np_arr = np.array(im)  # convert the image to a numpy array

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
        # plt.figure(1)
        # plt.subplot(121)
        # plt.imshow(segmentation[0])
        # plt.axis('off')
        # plt.title('Segmentaion 1')

        # plt.subplot(122)
        # plt.imshow(segmentation[1])
        # plt.axis('off')
        # plt.title('Segmentaion 2')

        # # plot output_np_arr, which is a numpy array
        # plt.figure(2)
        # plt.imshow(output_np_arr)
        # plt.axis('off')
        # plt.title('Pose Prediction')

        plt.show()
        # plt.close(0)
        return pose_pred, output_np_arr

    def pose_matrix_to_euler(self, pose_matrix):
        from scipy.spatial.transform import Rotation
        rotation_matrix = pose_matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('zyx', degrees=True)
        return euler_angles

    def visualize_train(self, output, batch):
        import torch
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        # mask = batch['mask'][0].detach().cpu().numpy()
        mask = output['seg'][0][0].detach().cpu().numpy()
        # vertex = batch['vertex'][0][0].detach().cpu().numpy()
        vertex = output['vertex'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
        # fps_2d = output['kpt_2d'][0].detach().cpu().numpy()
        plt.figure(0)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.subplot(223)
        plt.imshow(vertex)
        plt.subplot(224)
        plt.imshow(inp)
        plt.scatter(fps_2d[:, 0], fps_2d[:, 1], color='red', s=10)
        plt.savefig('test.jpg')
        

        # # Assuming 'output' is your PyTorch tensor
        # output = output['vertex'][0].detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array

        # # Extract vectors from the array
        # vectors = output[1] - output[0]  # Compute vectors relative to the first one (output[0])

        # # Extract x and y components of the vectors
        # x_components = vectors[:, 0]
        # y_components = vectors[:, 1]

        # # Create a figure and axis
        # plt.figure(1)
        # fig, ax = plt.subplots()

        # # Plot arrows for each vector
        # for i, (x, y) in enumerate(zip(x_components, y_components)):
        #     ax.arrow(output[0, 1], output[0, 2], x, y, head_width=5, head_length=5, fc='blue', ec='blue', label=f'Keypoint {i + 1}')
        # # Set axis limits and labels
        # ax.set_xlim(output[0, 1] - 10, output[0, 1] + 10)
        # ax.set_ylim(output[0, 2] - 10, output[0, 2] + 10)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # # Add legend
        # ax.legend()
        

        plt.close(0)

    def visualize_gt(self, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)

        print("Image id: \n", img_id)
        print("keypoints in 3d: \n", kpt_3d)
        print("Intrinsic parameters: \n", K)
        print("Pose GT: \n", pose_gt)
        print("corner gt in 3d: \n", corner_3d)
        print("corner gt in 2d: \n", corner_2d_gt)

        _, ax = plt.subplots(1)
        ax.imshow(inp)
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        plt.show()


        # Not appliable because of gimbal lock
        # ###################################################
        # def average_error(pose_pred, pose_gt):
        #     from scipy.spatial.transform import Rotation

        #     translation_error = np.abs(pose_pred[:, 3] - pose_gt[:, 3])

        #     rotation_pred = Rotation.from_matrix(pose_pred[:, :3])
        #     rotation_gt = Rotation.from_matrix(pose_gt[:, :3])
        #     euler_error = rotation_pred.inv() * rotation_gt
        #     euler_error = euler_error.as_euler('zyx', degrees=True)
        #     euler_error = np.abs(euler_error)
        #     print(pose_gt)
        #     print(pose_pred)
        #     print('Translation Error (X-axis): {:.1f} mm'.format(translation_error[0] * 1000))
        #     print('Translation Error (Y-axis): {:.1f} mm'.format(translation_error[1] * 1000))
        #     print('Translation Error (Z-axis): {:.1f} mm'.format(translation_error[2] * 1000))

        #     print('Euler Angle Error (X-axis): {:.1f} deg'.format(euler_error[0]))
        #     print('Euler Angle Error (Y-axis): {:.1f} deg'.format(euler_error[1]))
        #     print('Euler Angle Error (Z-axis): {:.1f} deg'.format(euler_error[2]))

        #     euler_pred = rotation_pred.as_euler('zyx', degrees=True)
        #     euler_gt = rotation_gt.as_euler('zyx', degrees=True)
        #     print('Euler Angle (X-axis) - pred: {:.1f} deg'.format(euler_pred[0]))
        #     print('Euler Angle (Y-axis) - pred: {:.1f} deg'.format(euler_pred[1]))
        #     print('Euler Angle (Z-axis) - pred: {:.1f} deg'.format(euler_pred[2]))

        #     print('Euler Angle (X-axis) - gt: {:.1f} deg'.format(euler_gt[0]))
        #     print('Euler Angle (Y-axis) - gt: {:.1f} deg'.format(euler_gt[1]))
        #     print('Euler Angle (Z-axis) - gt: {:.1f} deg'.format(euler_gt[2]))
        # average_error(pose_pred, pose_gt)
        
        # ###################################################

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