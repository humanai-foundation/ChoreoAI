# Reference: https://github.com/mariel-pettee/choreo-graph/blob/7e50d643a2b09196ff08934d383c5f68815ddff2/functions/load_data.py
import numpy as np
import torch
from glob import glob
import os
import sys
sys.path.append('../')

# 53 points
point_labels = [
    'ARIEL', 'C7', 'CLAV', 'LANK', 'LBHD', 'LBSH', 'LBWT', 'LELB', 'LFHD', 'LFRM', 'LFSH', 'LFWT', 'LHEL', 'LIEL', 'LIHAND', 'LIWR', 'LKNE', 'LKNI', 'LMT1', 'LMT5', 'LOHAND', 'LOWR', 'LSHN', 'LTHI', 'LTOE', 'LUPA', 'MBWT', 'MFWT', 'RANK', 'RBHD', 'RBSH', 'RBWT', 'RELB', 'RFHD', 'RFRM', 'RFSH', 'RFWT', 'RHEL', 'RIEL', 'RIHAND', 'RIWR', 'RKNE', 'RKNI', 'RMT1', 'RMT5', 'ROHAND', 'ROWR', 'RSHN', 'RTHI', 'RTOE', 'RUPA', 'STRN', 'T10'
]

reduced_joint_names = [
    'ARIEL', 'CLAV', 'RFSH', 'LFSH', 'RIEL', 'LIEL', 'RIWR', 'LIWR', 'RKNE', 'LKNE', 'RTOE', 'LTOE', 'LHEL', 'RHEL', 'RFWT', 'LFWT', 'LBWT', 'RBWT'
]

skeleton_lines = [
    #  ( (start group), (end group) ),
    (('LHEL',), ('LTOE',)), # toe to heel
    (('RHEL',), ('RTOE',)),
    (('LMT1',), ('LMT5',)), # horizontal line across foot
    (('RMT1',), ('RMT5',)),   
    (('LHEL',), ('LMT1',)), # heel to sides of feet
    (('LHEL',), ('LMT5',)),
    (('RHEL',), ('RMT1',)),
    (('RHEL',), ('RMT5',)),
    (('LTOE',), ('LMT1',)), # toe to sides of feet
    (('LTOE',), ('LMT5',)),
    (('RTOE',), ('RMT1',)),
    (('RTOE',), ('RMT5',)),
    (('LKNE',), ('LHEL',)), # heel to knee
    (('RKNE',), ('RHEL',)),
    (('LFWT',), ('RBWT',)), # connect pelvis
    (('RFWT',), ('LBWT',)), 
    (('LFWT',), ('RFWT',)), 
    (('LBWT',), ('RBWT',)),
    (('LFWT',), ('LBWT',)), 
    (('RFWT',), ('RBWT',)), 
    (('LFWT',), ('LTHI',)), # pelvis to thighs
    (('RFWT',), ('RTHI',)), 
    (('LBWT',), ('LTHI',)), 
    (('RBWT',), ('RTHI',)), 
    (('LKNE',), ('LTHI',)), 
    (('RKNE',), ('RTHI',)), 
    (('CLAV',), ('LFSH',)), # clavicle to shoulders
    (('CLAV',), ('RFSH',)), 
    (('STRN',), ('LFSH',)), # sternum & T10 (back sternum) to shoulders
    (('STRN',), ('RFSH',)), 
    (('T10',), ('LFSH',)), 
    (('T10',), ('RFSH',)), 
    (('C7',), ('LBSH',)), # back clavicle to back shoulders
    (('C7',), ('RBSH',)), 
    (('LFSH',), ('LBSH',)), # front shoulders to back shoulders
    (('RFSH',), ('RBSH',)), 
    (('LFSH',), ('RBSH',)),
    (('RFSH',), ('LBSH',)),
    (('LFSH',), ('LUPA',),), # shoulders to upper arms
    (('RFSH',), ('RUPA',),), 
    (('LBSH',), ('LUPA',),), 
    (('RBSH',), ('RUPA',),), 
    (('LIWR',), ('LIHAND',),), # wrist to hand
    (('RIWR',), ('RIHAND',),),
    (('LOWR',), ('LOHAND',),), 
    (('ROWR',), ('ROHAND',),),
    (('LIWR',), ('LOWR',),), # across the wrist 
    (('RIWR',), ('ROWR',),), 
    (('LIHAND',), ('LOHAND',),), # across the palm 
    (('RIHAND',), ('ROHAND',),), 
    (('LFHD',), ('LBHD',)), # draw lines around circumference of the head
    (('LBHD',), ('RBHD',)),
    (('RBHD',), ('RFHD',)),
    (('RFHD',), ('LFHD',)),
    (('LFHD',), ('ARIEL',)), # connect circumference points to top of head
    (('LBHD',), ('ARIEL',)),
    (('RBHD',), ('ARIEL',)),
    (('RFHD',), ('ARIEL',)),
]


def load_data(pattern="data/mariel_*.npy"):
    datasets = {}
    ds_all = []

    exclude_points = [26, 53]
    point_mask = np.ones(55, dtype=bool)
    point_mask[exclude_points] = 0

    for f in sorted(glob(pattern)):
        # [joints, timestep, dimension]
        ds_name = os.path.basename(f)[7: -4]
        # [timestep, joints, dimension]
        ds = np.load(f).transpose((1, 0, 2))
        # delete first 500 and last 500 timesteps
        ds = ds[500: -500, point_mask]
        ds[:, :, 2] *= -1
        # betternot_and_retrograde: 9925; beyond: 5803; chunli: 3866; honey: 5309; knownbetter: 6649; penelope: 6757
        datasets[ds_name] = ds
        ds_all.append(ds)
    
    ds_counts = np.array([ds.shape[0] for ds in ds_all])
    ds_offsets = np.zeros_like(ds_counts)
    ds_offsets[1:] = np.cumsum(ds_counts[: -1])
    ds_all = np.concatenate(ds_all)

    print("Original numpy dataset contains {:,} timesteps of {} joints with {} dimensions each.".format(ds_all.shape[0], ds_all.shape[1], ds_all.shape[2]))

    low, hi = np.quantile(ds_all, [0.01, 0.99], axis=(0, 1))
    xy_min = min(low[:2])
    xy_max = max(hi[:2])
    xy_range = xy_max - xy_min

    # normalize data
    ds_all[:, :, :2] -= xy_min
    ds_all *= 2 / xy_range
    ds_all[:, :, :2] -= 1.0

    ds_all_centered = ds_all.copy()
    # center data
    ds_all_centered[:, :, :2] -= ds_all_centered[:, :, :2].mean(axis=1, keepdims=True)

    datasets_centered = {}

    for ds in datasets:
        datasets[ds][:, :, :2] -= xy_min
        datasets[ds] *= 2 / xy_range
        datasets[ds][:, :, :2] -= 1.0
        datasets_centered[ds] = datasets[ds].copy()
        datasets_centered[ds][:, :, :2] -= datasets[ds][:, :, :2].mean(axis=1, keepdims=True)
    
    # calculate velocity for each frame
    velocities = np.vstack([np.zeros((1, 53, 3)), np.array([35 * (ds_all[t + 1, :, :] - ds_all[t, :, :]) for t in range(len(ds_all) - 1)])])

    ds_all = np.dstack([ds_all, velocities])
    ds_all_centered = np.dstack([ds_all_centered, velocities])

    for data in [ds_all, ds_all_centered]:
        # Normalize locations & velocities (separately) to [-1, 1]
        loc_min = np.min(data[:, :, :3])
        loc_max = np.max(data[:, :, :3])
        vel_min = np.min(data[:, :, 3:])
        vel_max = np.max(data[:, :, 3:])
        data[:, :, : 3] = (data[:, :, :3] - loc_min) * 2 / (loc_max - loc_min) - 1
        data[:, :, 3:] = (data[:, :, 3:] - vel_min) * 2 / (vel_max - vel_min) - 1

    # [38309, 53, 6]
    return ds_all, ds_all_centered, datasets, datasets_centered, ds_counts


def edges(reduced_joints, seq_len):
    # Define a subset of joints if we want to train on fewer joints that still capture meaningful body movement
    if reduced_joints:
        reduced_joint_indices = [point_labels.index(joint_name) for joint_name in reduced_joint_names]
        edge_index = np.array([(i, j) for i in reduced_joint_indices for j in reduced_joint_indices if i != j])
    else:
        reduced_joint_indices = None
        edge_index = np.array([(i, j) for i in range(53) for j in range(53) if i != j]) # note: no self-loops!

    skeleton_idxs = []
    for g1, g2 in skeleton_lines:
        entry = []
        entry.append([point_labels.index(l) for l in g1][0])
        entry.append([point_labels.index(l) for l in g2][0])
        skeleton_idxs.append(entry)
        
    is_skeleton_edge = []      
    for edge in np.arange(edge_index.shape[0]): 
        if [edge_index[edge][0], edge_index[edge][1]] in skeleton_idxs: 
            is_skeleton_edge.append(torch.tensor(1.0))
        else:
            is_skeleton_edge.append(torch.tensor(0.0))
    
    is_skeleton_edge = np.array(is_skeleton_edge)
    copies = np.tile(is_skeleton_edge, (seq_len, 1)) # create copies of the 1D array for every timestep
    skeleton_edges_over_time = torch.tensor(np.transpose(copies))
    
    if reduced_joints: 
        ### Need to remake these lists to include only nodes 0-18 now
        edge_index = np.array([(i, j) for i in np.arange(len(reduced_joint_indices)) for j in np.arange(len(reduced_joint_indices)) if i != j])
    
    return torch.tensor(edge_index, dtype=torch.long), skeleton_edges_over_time, reduced_joint_indices
