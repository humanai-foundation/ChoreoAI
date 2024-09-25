from glob import glob

import numpy as np
import torch

from utils import *


def load_data(seq_len, frame_gap=1, n_sample_dancers=None):
    print('######## Array Information Coming From Original Videos ########')

    # Reading data coming from the pre-processing pipeline and creating both dancers
    joint_poses = []
    for file in glob('./data/*.npy'):
        interleaved_poses = np.load(file)
        poses_1 = interleaved_poses[0::2]
        poses_2 = interleaved_poses[1::2]
        
        # Sampling frames for movement smoothness
        joint_poses.append(np.concatenate((poses_1, poses_2), axis=1)[::3])
        print('Joint poses {} shape: {}\n'.format(file.split('/')[-1], joint_poses[-1].shape))

    # Building initial transposed edge index (adjacencies)
    edge_index_t = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [7, 27], [8, 11],
                    [8, 28], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17], [15, 24], [16, 18], [17, 19],
                    [18, 20], [19, 21], [20, 22], [21, 23], [22, 25], [23, 26]]

    # Getting second person skeleton
    n_joints = int(joint_poses[0].shape[1]/2)
    init_skeleton_len = len(edge_index_t)
    for edge_index in range(init_skeleton_len):
        edge_index_t.append([edge_index_t[edge_index][0]+n_joints, edge_index_t[edge_index][1]+n_joints])

    # Saving skeletons for visualization
    skeletons = edge_index_t.copy()

    # Fully connecting the two people
    for joint_1 in range(interleaved_poses.shape[1]):
        for joint_2 in range(interleaved_poses.shape[1]):
            edge_index_t.append([joint_1, joint_2+n_joints])

    # Making graph undirected
    full_skeleton_len = len(edge_index_t)
    for edge_index in range(full_skeleton_len):
        edge_index_t.append([edge_index_t[edge_index][1], edge_index_t[edge_index][0]])

    print(f'######## Total Number of Edges: {len(edge_index_t)} ########\n')

    ########## REMOVED PIECE OF CODE. NOW WE IMPLEMENT ROTATION WITHIN BATCH PROCESSING ##########
    # # Augmenting poses by adding random rotations
    # rand_rot_num = 4
    # joint_poses_augmented = [joint_poses]
    # for r in range(rand_rot_num):
    #     joint_poses_rot = []
        
    #     angle_x = 2*np.random.rand(1)[0]*np.pi
    #     angle_y = 2*np.random.rand(1)[0]*np.pi
    #     angle_z = 2*np.random.rand(1)[0]*np.pi
        
    #     for choreo in joint_poses:
    #         choreo_aux = []
            
    #         for frame in choreo:
    #             rotated_points = rotate_points(frame, angle_x, angle_y, angle_z)    
    #             choreo_aux.append(rotated_points)
                
    #         joint_poses_rot.append(np.array(choreo_aux))

    #     joint_poses_augmented.append(joint_poses_rot)


    # Estimating velocity of points
    velocities = []
    joint_poses_velo = []
    for joint_pose in [joint_poses]:
        velocities_aux = []
        joint_poses_velo_aux = []
        
        for choreo in joint_pose:
            choreo = torch.Tensor(choreo)
        
            velocity_choreo = compute_velocities(choreo, frame_gap)
            velocities_aux.append(velocity_choreo)
            
            joint_poses_velo_aux.append(torch.cat((choreo, velocity_choreo), dim=-1))

        velocities.append(velocities_aux)
        joint_poses_velo.append(joint_poses_velo_aux)

    batches = []
    choreo_lens = []

    # # Building non-overlapping sequences
    # for joint_pose_velo in joint_poses_velo:
    #     for choreo in joint_pose_velo:
    #         choreo = torch.Tensor(choreo)
        
    #         num_seqs = choreo.shape[0] // seq_len
    #         batches.append(torch.stack([choreo[i*seq_len:(i+1)*seq_len] for i in range(num_seqs)]))
    #         choreo_lens.append(batches[-1].size(0))

    # Building overlapping sequences
    for joint_pose_velo in joint_poses_velo:
        for choreo in joint_pose_velo:
            choreo = torch.Tensor(choreo)
        
            batches.append(torch.stack([choreo[i:i+seq_len] for i in range(len(choreo)-seq_len)]))
            choreo_lens.append(batches[-1].size(0))

    batches = torch.cat(batches, dim=0)

    # Fixing x, y, z configuration
    batches[:, :, :, [0, 1, 2]] = batches[:, :, :, [2, 0, 1]]
    batches[:, :, :, 2] = -batches[:, :, :, 2]

    # Balanced training-validation split
    train_split = []
    val_split = []
    train_batches = []
    val_batches = []
    split_percentage = 0.85
    next_choreo = 0
    for choreo_len in choreo_lens:
        
        train_split.append(int(split_percentage*choreo_len))
        val_split.append(choreo_len - train_split[-1])
        
        train_batches.append(batches[next_choreo : next_choreo + train_split[-1]])
        val_batches.append(batches[next_choreo + train_split[-1] : next_choreo + choreo_len])
        
        next_choreo += choreo_len

    train_batches = torch.cat(train_batches, dim=0)
    val_batches = torch.cat(val_batches, dim=0)

    # Sampling joints and selecting edges connected to these joints if duet simplification requested
    if n_sample_dancers is not None:
        sampled_joints_1 = np.random.choice(n_joints, n_sample_dancers)
        sampled_joints_2 = np.random.choice(n_joints, n_sample_dancers) + n_joints
        sampled_joints = np.concatenate([sampled_joints_1, sampled_joints_2])
        print("Sampled joints for dancer 1: {}, and dancer 2: {}\n".format(sampled_joints_1, sampled_joints_2))

        # Mapping sampled joints to new indices for the message passing matrices
        edge_mapping = {e: c for c, e in zip(range(len(sampled_joints)), sampled_joints)}        

        sampled_batches = batches[:, :, sampled_joints, :]
        sampled_train_batches = train_batches[:, :, sampled_joints, :]
        sampled_val_batches = val_batches[:, :, sampled_joints, :]

        sampled_edge_index_t = []
        for (start, end) in edge_index_t[:len(edge_index_t)]:
            if (start not in sampled_joints) or (end not in sampled_joints):
                continue

            sampled_edge_index_t.append([edge_mapping[start], edge_mapping[end]])

        print(f'######## Total Number of Sampled Edges: {len(sampled_edge_index_t)} ########\n')

        print('######## Data Structures Created For Training ########\n')

        # Printing all the data structures created
        print('Shape of tensor with all sequences: {}'.format(sampled_batches.shape))
        print('Length of each choreography: {}\n'.format(choreo_lens))

        print('Shape of training data with all sequences: {}'.format(sampled_train_batches.shape))
        print('Length of each choreography in training dataset: {}\n'.format(train_split))

        print('Shape of validation data with all sequences: {}'.format(sampled_val_batches.shape))
        print('Length of each choreography in validation dataset: {}\n'.format(val_split))

        return joint_poses, sampled_edge_index_t, sampled_batches, choreo_lens, sampled_train_batches, train_split, sampled_val_batches, val_split

    print('######## Data Structures Created For Training ########\n')

    # Printing all the data structures created
    print('Shape of tensor with all sequences: {}'.format(batches.shape))
    print('Length of each choreography: {}\n'.format(choreo_lens))

    print('Shape of training data with all sequences: {}'.format(train_batches.shape))
    print('Length of each choreography in training dataset: {}\n'.format(train_split))

    print('Shape of validation data with all sequences: {}'.format(val_batches.shape))
    print('Length of each choreography in validation dataset: {}\n'.format(val_split))

    return joint_poses, edge_index_t, batches, choreo_lens, train_batches, train_split, val_batches, val_split