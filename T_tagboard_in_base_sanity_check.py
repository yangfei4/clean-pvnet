#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
# %matplotlib notebook
import matplotlib.pyplot as plt

import pytransform3d.plot_utils as plot_utils
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import transform_from, plot_transform


inch2m = 0.0254
adpaterplate_thickness = 1 * inch2m
tagboard_thickness = 0.125 * inch2m
Z_OFFSET_TAG_IN_BASE = -0.875 * inch2m # 1" adapter plate - 0.125" tagboard thickness
theta= np.pi / 2
opb_length = 914 / 1000
opb_width = 1524 / 1000
opb_height = 58 / 1000

R_breadboard_in_base = np.array([[np.cos(theta), np.sin(theta), 0],
								 [-np.sin(theta), np.cos(theta), 0],
								 [0, 0, 1]])

T_breadboard_in_base = np.block([np.eye(4,3) @ R_breadboard_in_base @ np.eye(3, 4) @ np.eye(4, 3), np.array([-opb_length / 2, 0, -adpaterplate_thickness, 1]).reshape(4,-1)])

def align_T_tagboard(T_tagboard_in_base):
    T_tagboard_aligned = inv_T(T_tagboard_in_base) @ T_tagboard_in_base
    R, t = T_tagboard_in_base[:3, :3], T_tagboard_in_base[:3, 3]
    t[2] = Z_OFFSET_TAG_IN_BASE
    return np.block([[R.T @ R, t.reshape(3, 1)],
                       [np.array([0, 0, 0, 1])]])

def get_ang_err(R_tag):
    # Calculate the angular difference between two rotation matrices θ = arccos((trace(R) - 1) / 2)
    ang_err = np.arccos((np.trace(np.eye(3) @ R_tag.T) - 1) / 2)
    ang_err *= 180/np.pi
    return ang_err


def inv_T(T):
    R_inv = T[:3, :3].T
    t_inv = R_inv @ T[:3, 3]
    return np.block([[R_inv, t_inv.reshape(3, 1)],
                     [np.array([0, 0, 0, 1])]])

def draw_workspace(T_camera_in_base, T_tagboard_in_base, figsize=(10,10)):
    # Visualize each frame
    plt.figure(figsize=figsize, constrained_layout=True)
    ax = make_3d_axis(1, 111, n_ticks=10, unit="m")
    ax.set_zticks([-0.3, 2])
    #ax.view_init(elev=30, azim=-70)

    T_tagboard_aligned = align_T_tagboard(T_tagboard_in_base)
    plot_transform(ax, s=0.5,  name="Robot Base")
    plot_transform(ax, s=0.5, A2B=T_camera_in_base, name="Camera in base")
    plot_transform(ax, s=0.25, A2B=T_tagboard_in_base, name="Tagboard in base")
    # plot_transform(ax, s=0.5, A2B=T_breadboard_in_base, name="Breadboard in base")
    plot_transform(ax, s=0.5, A2B=T_tagboard_aligned, name="Corrected Tagboard in base")

    # Visualize x-y planes of each frame
    plot_utils.plot_box(ax, size=np.array([0.20, 0.20, 0.00254]), A2B=T_tagboard_in_base, color="purple", wireframe=False, ax_s=0.5, alpha=0.8)
    plot_utils.plot_box(ax, size=np.array([0.205, 0.205, 0.0254]), A2B=T_tagboard_aligned, color="yellow", wireframe=False, ax_s=1, alpha=0.6)
    plot_utils.plot_box(ax, size=np.array([0.20, 0.20, 0.0254]), color="green", wireframe=False, ax_s=0.5, alpha=0.8)
    plot_utils.plot_box(ax, A2B=T_breadboard_in_base, size= np.array([opb_length, opb_width, opb_height]), color="grey", wireframe=False, alpha=0.2)
    ax.set_title("Frames in Worksace")
    ax.legend(["Tagboard", "Robot base"])

    plt.show()
    
    
def plot_plane_intersections():
    plt.figure(figsize=(16, 16), constrained_layout=True)
    ax = make_3d_axis(1, 111, n_ticks=10, unit="m")
    ang_err = get_ang_err(R_tagboard_in_base)
    print(f"Rotational Angular Error between tagboard and robot base: {ang_err:.6f}")
    print(f"Tagboard aligned to Robot base: {np.allclose(ang_err, np.zeros(0))}")
    ax = make_3d_axis(1, 132, n_ticks=10, unit="m")
    plot_utils.plot_box(ax, size=np.array([1.205, 1.205, 0.0254]), A2B=T_tagboard_in_base, color="purple", wireframe=False, ax_s=1, alpha=0.6)
    plot_utils.plot_box(ax, size=np.array([1.20,  1.20,  0.0254]), color="green", wireframe=False, ax_s=0.5, alpha=0.8)
    ax.set_title(r"$T_{tagboard}^{base}$ X-Y Plane Visualized" + f"\n" +r"$R_{tagboard}^{base}$ Angular Error w.r.t $R_{base}$: " + f"{ang_err:.6f}" + r"$\degree$")
    ax.legend(["Tagboard", "Robot base"])
    
    
    
    ang_err = get_ang_err(R_tagboard_aligned)
    print(f"Rotational Angular Error between tagboard and robot base: {ang_err:.6f}")
    print(f"Tagboard aligned to Robot base: {np.allclose(ang_err, np.zeros(0))}")
    T_tagboard_corrected_with_trans_offset = np.block([np.eye(4, 3) @ R_tagboard_aligned @ np.eye(3, 4) @ np.eye(4, 3), np.array([0, 0.5, 0, 1]).reshape(4, -1)])
    ax = make_3d_axis(1, 133, n_ticks=10, unit="m")
    T_tag_in_base = plot_utils.plot_box(ax, size=np.array([1.205, 1.205, 0.0254]), A2B=T_tagboard_corrected_with_trans_offset, color="yellow", wireframe=False, ax_s=1, alpha=0.6)
    T_base = plot_utils.plot_box(ax, size=np.array([1.20,  1.20,  0.0254]), color="green", wireframe=False, ax_s=0.5, alpha=0.8)
    ax.set_title(r"Corrected $T_{tagboard}^{base}$ X-Y Plane Visualized" + f"\n" +r"$R_{tagboard}^{base}$ Angular Error w.r.t $R_{base}$: " + f"{ang_err:.6f}" + r"$\degree$")
    ax.legend(["Tagboard", "Robot base"])
    
    # plt.tight_layout()


# In[2]:


data_all_instances = np.load("./11_23_image_dataset/insertmold.npz", allow_pickle=True)
print("npz file keys: ", list(data_all_instances.keys()))
print()

T_tagboard_in_cam = np.load("./data/11_23_image_dataset/T_tagboard_in_cam.npy")
T_camera_in_base = np.load("./data/11_23_image_dataset/T_camera_in_base.npy")

T_base_in_cam  = np.linalg.inv(T_camera_in_base)
T_tagboard_in_base = T_camera_in_base @  T_tagboard_in_cam

draw_workspace(T_camera_in_base, T_tagboard_in_base, figsize=(8,8))


# In[3]:


# 1) Get T_part_in_base
instance_data = data_all_instances['1'].item()
instance_data['class']
T_part_in_cam = instance_data['T_part_in_cam']
T_part_in_base = T_camera_in_base @ T_part_in_cam

# 2) Construct a "virtual" tagboard frame wrt robot's base

T_tag_in_base = np.eye(4)
T_tag_in_base[0, 3] = 1
T_tag_in_base[1, 3] = 1
T_tag_in_base[2, 3] = Z_OFFSET_TAG_IN_BASE

# 3) Get T_part_in_tagboard
T_part_in_tag = np.linalg.inv(T_tag_in_base) @ T_part_in_base

# 4) Get T_stable_part_in_tagboard
from mrcnn.stable_poses_est import find_closest_stable_pose, construct_T_from_R_sta_and_T_est, stable_poses_R, alphas_insertmold, z_offsets_insertmold
R_part_in_tag = T_part_in_tag[:3, :3]
R_stable_part_in_tag = find_closest_stable_pose(stable_poses_R, R_part_in_tag)
T_stable_part_in_tag = construct_T_from_R_sta_and_T_est(R_stable_part_in_tag, T_part_in_tag, alphas_insertmold, z_offsets_insertmold)

# 5) Get T_stable_part_in_base
T_stable_part_in_base = T_tag_in_base @ T_stable_part_in_tag
print(f"T stable part in base\n{'='*50}\n",T_stable_part_in_base)




path_to_insert_mold = "/home/ham/GithubWorkspace/gui2.0/cad_models/insert_mold_nf.ply"

figsize=(10,10)
plt.figure(figsize=figsize, constrained_layout=True)
ax = make_3d_axis(1, 111, n_ticks=10, unit="m")

# Set the scaling factors for the x, y, and z axes
# x = 0.8
x = [-0.68, -0.58]
y = [-0.03, 0.03]
z=0.02
# Add Geometry
T_tagboard_aligned = align_T_tagboard(T_tagboard_in_base)
print(f"T aligned tagboard in base\n{'='*50}\n", T_tagboard_aligned)
plot_utils.plot_mesh(filename=path_to_insert_mold, A2B=T_stable_part_in_base, s=np.ones(3), alpha=0.8, ax_s=0.5)
# plot_utils.plot_box(ax, size=np.array([0.20, 0.20, tagboard_thickness/2]), A2B=T_tagboard_aligned, color="yellow", wireframe=False, ax_s=0.5, alpha=0.8)
plot_utils.plot_box(ax, size=np.array([8 * inch2m, 8 * inch2m, tagboard_thickness/2]), A2B=T_tagboard_aligned, color="yellow", wireframe=False, ax_s=0.5, alpha=0.2)
# plot_utils.plot_box(ax, size=np.array([0.20, 0.20, adpaterplate_thickness/2]), color="green", wireframe=False, ax_s=0.5, alpha=0.8)

# TODO(ham)
plot_transform(ax=ax,A2B=T_stable_part_in_base, s=0.5, lw=1)
plot_transform(ax=ax, A2B=T_tagboard_aligned, s=0.5, lw=1)
# plot_transform(ax=ax, name="Robot Base",s=0.5, lw=1)
ax.autoscale()
ax.auto_scale_xyz(x, y, [-.05, z])


plt.show()

