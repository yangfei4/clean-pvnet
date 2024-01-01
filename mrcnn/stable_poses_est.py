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


def construct_T_from_R_sta_and_T_est(R_sta, T_est, alphas, z_offsets):
    alpha_sta = Rotation.from_matrix(R_sta).as_euler('xyz', degrees=True)[0]
    index = np.where(alphas == int(alpha_sta))[0]
    z_offset_sta = z_offsets[index]
    t_sta = np.array([T_est[0,3], T_est[1,3], z_offset_sta], dtype=object)
    T_sta = construct_T(R_sta, t_sta)
    return T_sta


######################################
# NOTE: Add to a config or something #
######################################
# these two lists should be element-aligned. 
# i.e. alphas[i] should correspond to z_offsets[i]
alphas_insertmold = np.array([0, 12, -146, 156])
z_offsets_insertmold = np.array([0.0021, 0.0019, 0.0032, 0.0013])

stable_poses_R = construct_R_sta_list(alphas_insertmold)
"""
print("Number of stable poses to consider: ", len(stable_poses_R))


R_sta_in_tag = find_closest_stable_pose(stable_poses_R, T_est_in_tag[:3, :3])
T_sta_in_tag = construct_T_from_R_sta_and_T_est(R_sta_in_tag, T_est_in_tag, alphas_insertmold, z_offsets_insertmold)
print("Transformation Matrix for Estimated Pose:")
print(T_est_in_tag)
print("Euler angles for Estimated Pose:", Rotation.from_matrix(T_est_in_tag[:3, :3]).as_euler('xyz', degrees=True))
print()
print("Transformation Matrix for Stable Pose:")
print(T_sta_in_tag)
print("Euler angles for closest stable pose:", Rotation.from_matrix(T_sta_in_tag[:3, :3]).as_euler('xyz', degrees=True))

print()
print("Angular Difference: ", angular_R_diff(T_est_in_tag[:3, :3], T_sta_in_tag[:3, :3]), "degrees")
"""
