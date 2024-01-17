"""
Purpose: This script contains helper functions for the brute-force process of taking a pose estimate from PVNet and approximating the nearest stable pose.
Author: Yangfei Dai (yangfei4@illinois.edu)
"""
from scipy.spatial.transform import Rotation
import numpy as np

def construct_R_sta_list(alphas):
    beta = 0
    # decretize gama at step of 1 degree
    gamas = np.arange(0, 360, 1)
    stable_poses_R = []
    for alpha in alphas:
        for gama in gamas:
            R = Rotation.from_euler('xyz', [alpha, beta, gama], degrees=True)
            stable_poses_R.append(R.as_matrix())
    return stable_poses_R


def angular_R_diff(R1, R2):
    R_diff = R1.T @ R2
    trace = np.trace(R_diff)
    angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
    return angular_distance

def find_closest_stable_pose(stable_poses_R, R_est):
    angular_diffs = []
    for R_sp in stable_poses_R:
        angular_diffs.append(angular_R_diff(R_sp, R_est))

    # Get indices of the top three smallest angular differences
    top_index = np.argsort(angular_diffs)[0]
    # Extract the corresponding rotation matrices
    closest_stable_poses_R = stable_poses_R[top_index]
    return closest_stable_poses_R

def construct_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def construct_T_stable_from_T_est(T_est, cls: int):
    _, alphas, z_offsets = cls_alpha_zoffset_map[cls]
    stable_poses_R = construct_R_sta_list(alphas)

    R_sta = find_closest_stable_pose(stable_poses_R, T_est[:3, :3])
    alpha_sta = Rotation.from_matrix(R_sta).as_euler('xyz', degrees=True)[0]

    index = np.where(alphas == int(alpha_sta))[0]
    z_offset_sta = z_offsets[index]

    t_sta = np.array([T_est[0,3], T_est[1,3], z_offset_sta], dtype=object)
    T_sta = construct_T(R_sta, t_sta)
    return T_sta


######################################
# NOTE: Add to a config or something #
######################################
# See reference: https://uofi.app.box.com/integrations/googledss/openGoogleEditor?fileId=1404160811170&trackingId=3&csrfToken=c281b3dfc792e2a5d3def01e3e4385dc49f04277418bf2a4a8e97d95665d07c5#slide=id.g2ad96180aa3_2_40
# these two lists should be element-aligned. 
# i.e. alphas[i] should correspond to z_offsets[i]
alphas_mainshell  = np.array([0, 180])
alphas_topshell   = np.array([0, 148, 180])
alphas_insertmold = np.array([0, 12, 156])

z_offsets_mainshell  = np.array([0.0015, 0.0015])
z_offsets_topshell   = np.array([0.0022, 0.0026, 0.0022])
z_offsets_insertmold = np.array([0.0021, 0.0019, 0.0013])

cls_alpha_zoffset_map = {0: ("mainshell" , alphas_mainshell , z_offsets_mainshell),
                         1: ("topshell"  , alphas_topshell  , z_offsets_topshell),
                         2: ("insertmold", alphas_insertmold, z_offsets_insertmold)}

