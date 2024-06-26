from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg as default_cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
import cv2
import torch

from cobot_pipeline import reproject_keypoints

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

        # draw bounding box
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
    
        plt.show()
        return fig

    def vector_preprocess(self, mask, vertex):
        '''
        in our case, b(batch) size is 1
        :param mask:      [b,h,w]
        :param vertex:    [b,h,w,vn,2]
        :return: [coords,direct]
                 coords: [tn,2] - tn is the number of foreground pixels
                 direct: [tn,vn,2] - in each pixel, the vector to the vertex
        '''
        b, h, w, vn, _ = vertex.shape
        for bi in range(b):
            cur_mask = (mask[bi]).byte()

            coords = torch.nonzero(cur_mask).float()  # [tn,2]
            coords = coords[:, [1, 0]]
            # direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
            direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3).bool())  # [tn,vn,2]
            direct = direct.view([coords.shape[0], vn, 2])
        
        return coords, direct

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

    def draw_vector(self, output, output_path="/pvnet/data/visualization/inference"):
        mask = torch.argmax(output['seg'], 1)
        vertex = output['vertex'].permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex.shape
        vertex = vertex.view(b, h, w, vn_2//2, 2)
        coord, direct = self.vector_preprocess(mask, vertex)
        coord = coord.detach().cpu().numpy()
        direct = direct.detach().cpu().numpy()
        mask = mask[0].detach().cpu().numpy()

        # plot vectors for each keypoint
        from matplotlib.colors import hsv_to_rgb
        for i in range(direct.shape[1]):
            # plot vectors for each pixel on input image
            vector_map = np.zeros((128, 128, 3)).astype(np.float32)
            for j in range(direct.shape[0]):
                # import pdb; pdb.set_trace()
                x_idx = int(round(coord[j, 1]))
                y_idx = int(round(coord[j, 0]))

                # magnitude = np.linalg.norm(direct[j, i])
                magnitude = 1
                direction = np.arctan2(direct[j, i, 1], direct[j, i, 0])

                # Convert direction to hue value in the range [0, 2*pi]
                hue = (direction + np.pi) / (2 * np.pi)

                # Scale the magnitude to the range [0, 1] for saturation
                saturation = magnitude / np.max(np.linalg.norm(direct, axis=2))

                # Set the value (brightness) to 1
                value = 1

                # Convert HSV color to RGB
                color = hsv_to_rgb([hue, saturation, value])
                # Update the color of the vector in the vector map
                vector_map[x_idx, y_idx] = color

            vector_map = (vector_map*255).astype(np.uint8)
            plt.imshow(vector_map)
            plt.savefig(f'{output_path}/vectors_kpt{i}.png')

    def get_top4_vectors_indices(self, var):
        # Compute the trace of each covariance matrix
        traces = np.trace(var, axis1=-2, axis2=-1)
        # Get the indices that would sort the traces in ascending order
        sorted_indices = np.argsort(traces)

        # Select the top 4 indices
        top_4_indices = sorted_indices[:4]

        return top_4_indices

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


        self.draw_vector(output)

        if K_cam is None:
            K_cam = np.array([[10704.062350, 0, 2107+64], 
                      [0, 10727.438047, 1323+64],
                      [0, 0, 1]])

        img_id = int(batch_example['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        corner_3d = np.array(anno['corner_3d'])

        top4_kpts_index = self.get_top4_vectors_indices(output['var'][0].detach().cpu().numpy())
        # kpt_3d = kpt_3d[top4_kpts_index]
        # kpt_2d = kpt_2d[top4_kpts_index]
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K_cam)
        # pose_pred = pvnet_pose_utils.pnp(selected_kpt_3d, selected_kpt_2d, K_cam)

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

        plt.show()
        # plt.close(0)
        plt.savefig('/pvnet/data/visualization/inference/result.png')
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

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        ax[0, 0].imshow(inp)
        ax[0, 0].title("input image")

        ax[0, 1].imshow(mask)
        ax[0, 1].title("input image")
        
        ax[1, 0].imshow(vertex)

        ax[1, 1].imshow(inp)
        ax[1, 1].scatter(fps_2d[:, 0], fps_2d[:, 1], color='red', s=10)

        plt.savefig('test.jpg')
        return fig

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
        

        # plt.close(0)

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
    
    def get_image_and_tensor_for_batch(self, batch):
        img = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0) # (128,128,3)

        # plt.imshow(img)
        # plt.show()

        return img.numpy()

    def make_figure_for_training(self, input_img, pvnet_output, img_id):
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])
        kpt_2d = pvnet_output['kpt_2d'][0].detach().cpu().numpy()
        corner_3d = np.array(anno['corner_3d'])

        pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        segmentation = pvnet_output['seg'][0].detach().cpu().numpy()
        mask = pvnet_output['mask'][0].detach().cpu().numpy()
        mask_gt = cv2.imread(anno["mask_path"])[:, :, 0]
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)

        pred_reproject = reproject_keypoints(input_img.copy(), K, pose_pred, self.cfg)
        gt_reproject = reproject_keypoints(input_img.copy(), K, pose_gt, self.cfg)

        ###########################
        # overall result
        ###########################
        # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig, axs = plt.subplots(2, 3, figsize=(12, 12))

        # axs[0, 0].imshow(input_img)
        # axs[0, 0].scatter(kpt_2d[:8, 0], kpt_2d[:8, 1], color='red', s=10)
        axs[0, 0].imshow(mask)
        axs[0, 0].axis('off')
        axs[0, 0].set_title('Predicted Mask')

        axs[0, 1].imshow(pred_reproject)
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Predicted Keypoints')

        axs[0, 2].imshow(draw_axis(input_img.copy(), pose_pred[:3, :3], pose_pred[:3, 3], K))
        axs[0, 2].add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        axs[0, 2].add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))

        axs[0, 2].axis('off')
        axs[0, 2].set_title('Pose Prediction')

        axs[1, 0].imshow(mask_gt)
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Ground Truth Mask')

        axs[1, 1].imshow(gt_reproject)
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Ground Truth Keypoints')


        axs[1, 2].imshow(draw_axis(input_img.copy(), pose_gt[:3, :3], pose_gt[:3, 3], K))
        axs[1, 2].add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        axs[1, 2].add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        axs[1, 2].axis('off')
        axs[1, 2].set_title('Ground Truth Pose')
        fig.tight_layout()
        return fig


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

    return img
