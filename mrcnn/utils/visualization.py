from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pycocotools.coco import COCO
from torchvision.utils import make_grid


def plot_im(img: Union[str, Path, np.ndarray], figsize=(10,10), avoid_clr_swp: bool=False):
    
    if not isinstance(img, np.ndarray):
        assert Path(img).exists(), f"{img} is not a valid path"
        img = cv2.imread(str(img))
    
    if img.shape[-1] == 3 and not avoid_clr_swp:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig = plt.figure(figsize=figsize,tight_layout=True)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    fig.set_tight_layout(True)
    plt.imshow(img)
    
    
def plot_patches(patches, cols):
    from mpl_toolkits.axes_grid import ImageGrid
    
    from mpl_toolkits.axes_grid import ImageGrid
    p = patches.copy()
    if len(p.shape) != 4:
        _, _, H, W, C = p.shape
        p = p.reshape(-1, H, W, C)
    
    num_patches = p.shape[0]
    rows = num_patches // cols
    
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     share_all=True)
    
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

    for ax, im in zip(grid, p):
        if im.shape[-1] != 3:
            ax.imshow(im.sum(-1))
        else:
            ax.imshow(im)


def mask_with_clr(mask, clr) -> np.array:
    """
    :return np.ndarray: masked out image showing only where the provided clr is present 
    """
    _m = np.zeros(mask.shape)
    
    _clr = clr.copy()
    if _clr.size == 3:
        _clr =  np.concatenate((_clr, np.ones(1)))
    
    moi = np.all(mask == _clr, axis=-1)
    _m[moi] = _clr
    
    return _m

def compute_iou(i, u):
    return np.sum(i) / np.sum(u)

def generate_mask_reel(gt_m, p_m, clr, render: bool=False) -> Tuple[np.ndarray]:
    if clr.size == 3:
        alpha = np.ones(1)
        clr = np.concatenate((clr, alpha))
        
    if len(gt_m.shape) == 3:
        gt_m = mask_with_clr(gt_m, clr)
        p_m = mask_with_clr(p_m, clr)
    
        bg = [0, 0, 0, 0]
        m1 = np.all(gt_m != bg, axis=-1)
        m2 = np.all(p_m  != bg, axis=-1)
    elif len(gt_m.shape) == 2:
        clr = 1.0
        m1 = gt_m
        m2 = p_m
    
    u = np.logical_or(m1, m2)
    union = np.zeros(gt_m.shape)
    union[u] = clr

    i = np.logical_and(m1, m2)
    intersect = np.zeros(gt_m.shape)
    intersect[i] = clr


    d = np.logical_xor(m1, m2)
    diff = np.zeros(gt_m.shape)
    diff[d] = clr

    if render:
        plot_im(np.concatenate((gt_m, p_m, diff, intersect, union), axis=1))
      
    return (diff, intersect, union)


def generate_iou_and_diff(gt_m, p_m, clr, render: bool=False) -> Tuple[np.ndarray]:
    if clr.size == 3:
        # alpha = np.array([255])
        alpha = np.ones(1)
        clr = np.concatenate((clr, alpha))
        
    if len(gt_m.shape) == 4:
        gt_m = mask_with_clr(gt_m, clr)
        p_m = mask_with_clr(p_m, clr)
    
        bg = [0, 0, 0, 0]
        m1 = np.all(gt_m != bg, axis=-1)
        m2 = np.all(p_m  != bg, axis=-1)
    elif len(gt_m.shape) == 2:
        clr = 1.0
        m1 = gt_m
        m2 = p_m
    
    u = np.logical_or(m1, m2)
    union = np.zeros(gt_m.shape)
    union[u] = clr

    i = np.logical_and(m1, m2)
    intersect = np.zeros(gt_m.shape)
    intersect[i] = clr


    d = np.logical_xor(m1, m2)
    diff = np.zeros(gt_m.shape)
    diff[d] = clr

    iou = compute_iou(i, u)
    
    # iou = np.intersect1d(m1, m2).shape[0] / np.union1d(m1, m2).shape[0]
#     gt_iou = pycmask.iou( pycmask.frPyObjects(m1, *m1.shape), 
#                           pycmask.frPyObjects(m2, *m2.shape), 
#                           pyiscrowd=[0])

#     assert gt_iou == iou, f"GT IoU: {gt_iou:.4f} does not match computed value: {iou:.4f}"
    if render:
        plot_im(np.concatenate((gt_m, p_m, diff, intersect, union), axis=1))
      
    return ((diff, intersect, union),
            iou)


def color_and_combine_mask(masks: List[np.ndarray], clrs: Optional[list]=None):

    # if isinstance(clrs, None):
    #       from cobot_ds.globals import render_helper
    #       clrs = render_helper.colors

    # Color the masks with unique colors
    clr_masks = []
    
    if not isinstance(masks, list):
        if len(masks.shape) == 2:
            masks = [masks]
        
    if not isinstance(clrs, list):
        if len(clrs.shape) == 1:
            clrs = [clrs]
        
    for m, c in zip(masks, clrs):
        mask = np.zeros((*m.shape, 3))
        mask[m==1] = c
        clr_masks.append(mask)

    # Combine the masks into a single image
    clr_masks = np.stack(clr_masks).sum(0)

    # Add alpha channel to mask image 
    clr_alpha = np.ones((*clr_masks.shape[:-1], 1))
    bg = np.equal(clr_masks[:,:,1], 0)
    clr_alpha[bg] = 0
    clr_masks = np.concatenate((clr_masks, clr_alpha), 2)
    return clr_masks


def get_cent(ann):
    x1, y1, w, h = ann['bbox']
    return (int(x1 + w//2), int(y1 + h//2))

def load_img(img: Union[np.array, Path, str], add_alpha: bool):
    if not isinstance(img, np.ndarray):
        assert Path(img).exists(), f"{img} is not a valid path"
        img = cv2.imread(str(img))

    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img / 255
    if add_alpha:
          # Add alpha channel to img
        im_alpha = np.ones((*img.shape[:-1], 1)) #* 255
        img = np.concatenate((img, im_alpha), axis=2)
    return img 


def get_clrs_in_img(im):
    return np.unique(im.reshape(-1, 4), axis=0)

def view_clr(clr):
    if clr.size == 3:
        clr = np.concatenate((clr, np.ones(1)))
        
    im = np.ones((*g.shape[:-1], 4)) * clr
    plot_im(im)
    

def intense_overlay(bg, mask, a_bg=0.8, a_m=0.8):
    """
    # See https://en.wikipedia.org/wiki/Alpha_compositing
    a_o   = (a_bg + a_m * (1 - a_bg))
    _m    = mask * a_m * (1 - a_bg)
    c_o   = (bg * a_bg + _m) / a_o
    return np.uint8(c_o)
    """
    return bg  +  mask 



def get_num_of_pixs(im, query_clr=[255, 255, 255, 0]):
    if all(np.equal(query_clr, [255, 255, 255, 0])):
        res = np.all(im != query_clr, axis=-1)
    else:
        res = np.all(im == query_clr, axis=-1)
    return np.sum(res)


def crop_roi(im, cent, size) -> np.ndarray:
    x, y = cent
    x_off = slice(x - size, x + size)
    y_off = slice(y - size, y + size)
    
    if len(im.shape) == 2:
        return im[y_off, x_off]
    return im[y_off, x_off, :]


def get_tagboard_center_and_ideal_crop_size(cur_img_path: Union[str, Path], verbose: bool=False) -> Tuple[int, int]:
    """
    Given a path to a scene image in the dataset, this function calculates the tagboard center and projects it to the image plane 
    
    :param cur_img_path: (str or Path) path to the current scene
    :param verbose: (bool) if True print output of projections
    
    :return center_coords: (tuple)  
    """
    from cobot_ds.pose import get_ee_in_base
    from cobot_ds.globals import scene_manager

    def get_idx(img_path: str):
        cur_path = Path(img_path)
        scene_imgs = list(sorted(Path(cur_path).parent.glob('*.png')))
        return scene_imgs.index(Path(img_path))
    
    def get_robot_config_file(img_path: str):
        cur_path = Path(img_path)
        return list(sorted(Path(cur_path).parent.glob('*.txt')))[0]

    def make_T(rot: np.ndarray, X: float=0.0, Y: float=0.0, Z: float=0.0):
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = np.array([X, Y, Z]).T
        return T
    
    # Calculate center coordinates of the 8.27' x 8.27' tag board
    inch2cm = 2.54
    cm2meter = 1 / 100
    side_length = 8.27
    cent_coord = (side_length / 2) * inch2cm * cm2meter
    
    
    # Calculate inner square's corner coordinates
    tag_dim = 4.5 
    spacing = 0.25
    inner_diag = 6.11 / np.sqrt(2) * inch2cm
    c = tag_dim + 2 * spacing

    corners = [[c             , -c             ],
               [c + inner_diag, -c             ],
               [c             , -c - inner_diag],
               [c + inner_diag, -c - inner_diag]]

    corners = np.array(corners) * cm2meter
    zw = np.tile([0, 1], 4).reshape(-1, 2)
    corners = np.c_[corners, zw]

    # Pose of the tag board center in the tag board frame used during extrinsic calibration 
    T_tagboard_center_in_tag = np.eye(4)
    T_tagboard_center_in_tag[0, 3] = cent_coord 
    T_tagboard_center_in_tag[1, 3] = -cent_coord

    # Produce transform to offset the origin used in extrinsic calibration tag board's corner closets to tag 0
    R_chain_pupil = np.diag([1, -1, 1]) # Flip Y axis
    offset = np.array([1.5 * cm2meter, -14.5 * cm2meter, 0])  # Center of Tag 0
    offset += np.array([-2.5 * cm2meter, -2.5 * cm2meter, 0]) # Corner of board
    T_origin_in_tag =  make_T(R_chain_pupil, *offset)

    # Update the tag board's origin's relative pose and index the new tag board position coords
    P_center_in_tag = (T_origin_in_tag @ T_tagboard_center_in_tag)[:, 3]

    # Camera to Image projection matrix 
    P = scene_manager.K @ np.c_[np.eye(3), np.zeros((3,1))] 
    
    # Grab and sort the scene images and robot config file
    cur_idx = get_idx(cur_img_path)
    cur_joint_pose = get_robot_config_file(cur_img_path)

        
    # Get the current pose of the ee in the robot's base 
    T_ee_in_base_cur = get_ee_in_base(cur_joint_pose, cur_idx)

    # Rely on extrinsics to approximate the pose of the tag wrt the camera
    T_cur_tag_in_cam = scene_manager.T_base_cam @ T_ee_in_base_cur @ scene_manager.T_tag_in_ee

    # Project tag board center to camera frame
    P_tagboard_cent_in_cam = T_cur_tag_in_cam @ P_center_in_tag

    # Project tag board center to image frame
    P_tagboard_cent_in_img = P @ P_tagboard_cent_in_cam 
    P_tagboard_cent_in_img /= P_tagboard_cent_in_img[2]
    u, v, _ = P_tagboard_cent_in_img

    cir_cent = (int(u), int(v))

    # Project corners to img frame
    corner_pixs = []
    for c in corners:
        P_c = P @ T_cur_tag_in_cam @ T_origin_in_tag @ c
        P_c /= P_c[2]
        u, v, _ = P_c
        corner_pixs.append( (u, v) )
        
    corner_pixs = np.stack(corner_pixs).astype(int)
    c1, c2 = corner_pixs[0, 0], corner_pixs[1, 0]
    crop_size = np.max(corner_pixs, 0)[0] - np.min(corner_pixs, 0)[0]
    crop_size //= 2
    
    
    if verbose:
        # Render a circle where the pixel coordinates of the tag board center are.
        img = cv2.imread(str(cur_img_path))
        clr = (255, 0, 255)
        r = 60
        thickness = -1
        print(f"Suggested crop size: {crop_size}, Resulting res: {crop_size * 2}")

        print(f"Current image index: {cur_idx}")
        print(f"Cent Coords (img): {cir_cent}")
        cv2.circle(img, cir_cent, r // 4, clr, thickness) 
        
        
        for c in corner_pixs:
            cv2.circle(img, c, r, clr, thickness) 
        
        # plot_im(img)
        return img, cir_cent, crop_size

    return cir_cent, crop_size

def crop_image_around_tagboard_center(im: Union[str, np.ndarray], cent_coords: Tuple[int], crop_dim: int):
    
    if isinstance(im, str):
        im = cv2.imread(im)
    u, v = cent_coords
    u_slice = slice(u - crop_dim, u + crop_dim)
    v_slice = slice(v - crop_dim, v + crop_dim)
    return im[v_slice, u_slice]

def padding_img(img: np.array, new_size: Tuple, img_cent_in_new_dim: Tuple):
    old_img_height, old_img_width, channels = img.shape

    # create new image of desired size and color (blue) for padding
    new_image_width, new_image_height= new_size
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

    x_center, y_center = img_cent_in_new_dim

    # copy img image into center of result image
    y_slice = slice(int(y_center) - old_img_height//2, int(y_center) + old_img_height//2)
    x_slice = slice(int(x_center) - old_img_width //2, int(x_center) + old_img_width//2)
    result[y_slice, x_slice] = img

    return result


# TODO (ham): Fix this ;)

# from pycocotools.coco import COCO

# def overlay_im_with_mask(img_path: Union[str, Path, np.ndarray], coco: COCO, ann: Dict, figsize=(10,10)):
#     _m    = ann2mask(ann, coco)
#     img   = load_img(img_path, add_alpha=True)
#     masks = color_and_combine_mask(_m)
#     return intense_overlay(img, masks) 

# def plot_im_with_mask(img_path: Union[str, Path, np.ndarray], coco: COCO, ann: Dict, figsize=(10,10)):
#     img = overlay_im_with_mask(img_path, coco, ann, figsize)
#     plot_im(img, figsize)

def plot_im_from_coco(coco: Union[str, Path, COCO], img_name: Union[str, Path]=None, img_index: int=None,
                         img_ids: Optional[List]=None, path_to_coco: Union[str, Path, None]=None):
    """
    :returns: cur_img_info (dict) - information of the found image
    Given a coco object, this function parses and plots a single image within the coco file.
    """
    # Setup the a matplot figure without borders, ticks, etc
    fig = plt.figure(figsize=(10, 10),tight_layout=True)
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
    fig.set_tight_layout(True)
  
    # Initialize the cur_img_info to None and load coco file from path if necessary
    cur_img_info = None
    if not isinstance(coco, COCO):
        path_to_coco = str(coco)
        coco = COCO(coco)

    assert path_to_coco and Path(path_to_coco).exists(), (f"Path_to_coco should not be none and needs to exists" +
                                                          f"\nProvided path: {path_to_coco}" +
                                                          f"\nExists: {Path(path_to_coco).exists()}")


    # Load all of the images or a specified subset from provided list of img_ids
    if isinstance(img_ids, type(None)): 
        img_ids = coco.getImgIds()

    img_info = coco.loadImgs(img_ids)
  
    # Parse coco object to find cur_img_info based on either image name or image id
    if not isinstance(img_index, type(None)):
        cur_img_info = img_info[img_index]

    elif not isinstance(img_name, type(None)):
        img_name = Path(img_name).name
        for cur in coco.loadImgs():
            if img_name == cur['file_name']:
                cur_img_info = cur
    else:
        # If neither img_index or img_name are provided raise execption
        raise Exception("Expected either img_index or img_name to be provided")

    # Check to make sure cur_img_info is set to true
    assert not isinstance(cur_img_info, type(None)), "Cur_img_info is None"

    # Load annotations              
    catIds = coco.getCatIds()                        
    annIds = coco.getAnnIds(imgIds=cur_img_info['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)               

    relative_data_path = Path(path_to_coco).parent
    cur_img_path = str(relative_data_path / cur_img_info['file_name'])

    cur_im = cv2.imread(cur_img_path)
    cur_im = cv2.cvtColor(cur_im, cv2.COLOR_BGR2RGB)

    clrs = [np.random.randint(0, 255, 3) for _ in range(len(anns))]
    masks = [coco.annToMask(a) for a in anns]

    masks = [(m.reshape(1, *m.shape).repeat(3, axis=0)) for m in masks]


    clr_masks = []
    for m, c in zip(masks, clrs):
        m = m.transpose(1, 2, 0) * c
        clr_masks.append(m)


    clr_masks = np.stack(clr_masks).sum(0)
    plt.imshow(clr_masks / 255 * 0.2 + cur_im / 255 * 0.8)
  
    return cur_img_info

            
# def plot_patches_with_overlay(coco_file, num_samples=30):
#     from mpl_toolkits.axes_grid import ImageGrid
    
#     coco = COCO(coco_file)

#     imgIds = coco.getImgIds()
#     img_info = coco.loadImgs(imgIds)


#     # Save image patches and patchify function parameters for reproducibility
#     cur_patch_path = Path(coco_file).parent
#     cur_scene = cur_patch_path.name
#     patches_coco_path = str(cur_patch_path / f'{cur_scene}_coco_patches.json')
    
#     np.random.seed(42)
    
#     cols = 3
#     rows = num_samples // cols
    
#     fig = plt.figure(figsize=(20, 20))
#     grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                      nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
#                      axes_pad=0.1,  # pad between axes in inch.
#                      share_all=True)
    
#     grid[0].get_yaxis().set_ticks([])
#     grid[0].get_xaxis().set_ticks([])
    
#     plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

    
#     sample = np.random.randint(0, len(img_info), num_samples)
    
    
#     for ax, sample_idx in zip(grid, sample):
        
#         cur_img_info = img_info[sample_idx]
#         catIds = coco.getCatIds()                        
#         annIds = coco.getAnnIds(imgIds=cur_img_info['id'], catIds=catIds, iscrowd=None)
#         anns = coco.loadAnns(annIds)
        
#         cur_path = str(cur_patch_path / cur_img_info['file_name'])
        
#         cur_im = cv2.imread(cur_path)
#         cur_im = cv2.cvtColor(cur_im, cv2.COLOR_BGR2RGB)
    
#         clrs = [np.random.randint(0, 255, 3) for _ in range(len(anns))]
#         masks = [coco.annToMask(a) for a in anns]
        
#         masks = [(m.reshape(1, *m.shape).repeat(3, axis=0)) for m in masks]
        
        
#         clr_masks = []
#         for m, c in zip(masks, clrs):
#             m = m.transpose(1, 2, 0) * c
#             clr_masks.append(m)
        
        
#         clr_masks = np.stack(clr_masks).sum(0)
#         ax.imshow(clr_masks / 255 * 0.2 + cur_im / 255 * 0.8)

        
# np.random.seed(42)
# seeded_colors = np.random.randint(0, 256, (100, 3)) / 255


# def ann2mask(anns: List[dict], cocofile: COCO):
#     # Convert the ann from coco to bit masks
#     # NOTE: One can use either the gt or prediction coco object. 
#     # The specific annotation should be from the proper source (e.ggt/pred).
#     masks = [cocofile.annToMask(a) for a in anns]
#     # masks = [(m.reshape(1, *m.shape).repeat(3, axis=0)) for m in masks]
#     return masks



# def get_part(coco_obj: COCO, cat_id: int):
#     return coco_obj.cats[cat_id]['name']
    
# def _slow_produce_reference_file(gt_coco: COCO, pred_coco: COCO, data_path: Union[Path, str], crop_size: int=128, visualize: bool=False, **kwargs):
#     """
#     # Compute with very ineffecient personal IoU function (eta around 25 mins)
#     """
    
#     plt.close()
#     if isinstance(data_path, str):
#         data_path = Path(data_path)
        
#     E = COCOeval(gt_coco, pred_coco); # initialize CocoEval object
#     # E.params.imgIds = [749] # TODO (ham): remove
#     E.params.iouThrs = np.array([0.5])
#     E.evaluate();                # run per image evaluation
#     E.accumulate();              # accumulate per image results
#     E.summarize();               # display summary metrics of results

#     data = {'ious':[],
#             'gtAnnsIds': [],
#             'predAnnsIds': [],
#             'imgId': [],
#             'catId': []}
    
#     for cur_res in tqdm(E.evalImgs, desc='Calculating IoU for each known matching'):
        
#         if isinstance(cur_res, type(None)):
#             continue
        
#         try:
#             # Get the cur image, get the iou, and corresponding pred & gt annids.
#             gt_anns = gt_coco.loadAnns(cur_res['gtIds'])
#             pred2gt_order = cur_res['gtMatches'].astype(int).reshape(-1) - 1
#             p_ids = pred_coco.getAnnIds()
#             pred_anns = pred_coco.loadAnns(p_ids)
#             ordered_pred_anns = [pred_anns[idx] for idx in pred2gt_order]
#         except Exception as e:
#             raise NotImplementedError(e)
        
#         num_gts = len(cur_res['gtIds'])
#         num_dts = len(cur_res['dtIds'])

#         false_positives = num_gts < num_dts

#         if false_positives:
#             continue
#             raise NotImplementedError(f"[False Positive] #GTs: {num_gts:3d} | #DTs: {num_dts:3d} Finish Implementation")
#         # raise NotImplementedError(f"[False Negative] #GTs: {num_gts:3d} | #DTs: {num_dts:3d} Finish Implementation")
            
#         # Load the corresponding image
#         cur_img_id = cur_res['image_id']
    
#         # Convert the ann from coco to bit masks
#         _g_masks = ann2mask( gt_anns          , gt_coco   )
#         _p_masks = ann2mask( ordered_pred_anns, pred_coco )
        
        
#         # Get the bbox cent
#         _g_cents = [get_cent(a) for a in gt_anns]
#         _p_cents = [get_cent(a) for a in ordered_pred_anns]
        
#         # Crop mask around bbox cent
#         _g_rois = [crop_roi(g, c, crop_size) for g, c in zip(_g_masks, _g_cents)]
#         _p_rois = [crop_roi(p, c, crop_size) for p, c in zip(_p_masks, _p_cents)]
        
#         # Calculate IoU
#         reels, ious = zip(*[generate_iou_and_diff(g, p, clr, render=visualize) for g, p, clr in zip(_g_rois, _p_rois, seeded_colors)])
        
#         # Get corresponding gt/pred ids
#         gt_annIds = [a['id'] for a in gt_anns]
#         p_annIds  = [a['id'] for a in ordered_pred_anns]
        
    #     # Save info to dictionary
    #     data['imgId'].extend([cur_img_id] * len(gt_annIds))
    #     data['catId'].extend([cur_res['category_id']] * len(ordered_pred_anns))
    #     data['ious'].extend(ious)
    #     data['gtAnnsIds'].extend(gt_annIds)
    #     data['predAnnsIds'].extend(p_annIds)
    # return data, E, cur_res


def construct_sorted_gridview_of_scene(scene_idx: int) -> np.array:
    """
    Generate a single image that contains all the images in a given scene. The images are sorted in the order that they were captured in.
    """
    # Load and sort images in the scene of interest
    from cobot_ds.globals import scene_manager
    paths_to_scene_images = (Path(scene_manager.data_all_root) / f'scene_{scene_idx}').glob('*.png')
    paths_to_scene_images = sorted([str(p) for p in paths_to_scene_images])

    assert len(paths_to_scene_images) == 75, f"Length of loaded images ({len(paths_to_scene_images)}) is not 75!"

    
    # Downscale images to reduce memory consumption & compute and convert to torch
    downscale_by = 16

    imgs = [load_img(p, add_alpha=False)[::downscale_by, ::downscale_by, :] for p in paths_to_scene_images]
    imgs = [torch.from_numpy(img).permute(2, 0, 1)                          for img in imgs]

    # Construct grid of 15 images per rows 
    imgs_per_row = 15
    # imgs = [imgs[slice(stop - imgs_per_row, stop)] for stop in range(imgs_per_row, 75 + imgs_per_row, imgs_per_row)]
    imgs = torch.stack(imgs)

    grid = make_grid(imgs, nrow=imgs_per_row, normalize=True)
    return grid.permute(1, 2, 0).cpu().numpy()
