from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils


mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
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

        corner_3d = np.array(anno['corner_3d'])
        corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)

        fig, ax = plt.subplots(1)
        ax.imshow(inp)
        ax.axis("off")

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

        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
        plt.show()
        return fig

    def visualize_output(self, input, output):
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        K = np.array([[1.90856e+03, 0.00000e+00, 1.28000e+02],
                    [0.00000e+00, 1.90906e+03, 1.28000e+02],
                    [0.00000e+00, 0.00000e+00, 1.00000e+00]])

        # (TODO-Yangfei: update these matrix)
        # fpt_3d + center_3d
        kpt_3d = np.array([[-4.04200004e-03,  3.81499995e-03,  9.36000026e-04],
                           [ 3.95799987e-03,  3.81499995e-03,  9.35000018e-04],
                           [ 2.76400009e-03, -4.72499989e-03, -9.20000020e-04],
                           [-2.84800003e-03, -4.72499989e-03, -9.19000013e-04],
                           [ 4.03500022e-03,  5.50000004e-05, -1.42099999e-03],
                           [-4.11899993e-03,  5.50000004e-05, -1.42099999e-03],
                           [ 8.70999997e-04,  2.81700003e-03, -2.02000001e-03],
                           [-3.91999987e-04,  3.20799998e-03,  1.24500005e-03],
                           [-4.20000000e-05, -4.55000000e-04,  5.30000000e-05]])

        corner_3d = np.array([[-0.004242, -0.004725, -0.00202 ],
                              [-0.004242, -0.004725,  0.002126],
                              [-0.004242,  0.003815, -0.00202 ],
                              [-0.004242,  0.003815,  0.002126],
                              [ 0.004158, -0.004725, -0.00202 ],
                              [ 0.004158, -0.004725,  0.002126],
                              [ 0.004158,  0.003815, -0.00202 ],
                              [ 0.004158,  0.003815,  0.002126]])

        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
        # import pdb;pdb.set_trace()

        fig, ax = plt.subplots(1)
        ax.imshow(input)
        ax.axis("off")

        # Add patches for corner_2d_gt and corner_2d_pred
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))

        plt.show()
        return fig


    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = batch['mask'][0].detach().cpu().numpy()
        vertex = batch['vertex'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
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
        plt.close(0)

    def visualize_gt(self, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)

        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        K = np.array(anno['K'])

        pose_gt = np.array(anno['pose'])

        corner_3d = np.array(anno['corner_3d'])
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



