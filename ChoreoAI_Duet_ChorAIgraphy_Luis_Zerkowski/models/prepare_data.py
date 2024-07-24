from glob import glob

import numpy as np
import torch

def load_data(seq_len):
    print('######## Array Information Coming From Original Videos ########')

    # Reading data coming from the pre-processing pipeline and creating both dancers
    joint_poses = []
    for file in glob('./data/*.npy'):
        interleaved_poses = np.load(file)
        poses_1 = interleaved_poses[0::2]
        poses_2 = interleaved_poses[1::2]
        
        joint_poses.append(np.concatenate((poses_1, poses_2), axis=1))
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

    batches = []
    choreo_lens = []

    # Building non-overlapping sequences
    # for choreo in joint_poses:
    #     choreo = torch.Tensor(choreo)

    #     num_seqs = choreo.shape[0] // seq_len
    #     batches.append(torch.stack([choreo[i*seq_len:(i+1)*seq_len] for i in range(num_seqs)]))
    #     choreo_lens.append(batches[-1].size(0))

    # Building overlapping sequences
    for choreo in joint_poses:
        choreo = torch.Tensor(choreo)

        batches.append(torch.stack([choreo[i:i+seq_len] for i in range(len(choreo)-seq_len)]))
        choreo_lens.append(batches[-1].size(0))

    batches = torch.cat(batches, dim=0)

    # Balanced training-validation split
    train_split = []
    val_split = []
    train_batches = []
    val_batches = []
    next_choreo = 0
    for choreo_len in choreo_lens:
        
        train_split.append(int(0.85*choreo_len))
        val_split.append(choreo_len - train_split[-1])
        
        train_batches.append(batches[next_choreo : next_choreo + train_split[-1]])
        val_batches.append(batches[next_choreo + train_split[-1] : next_choreo + choreo_len])
        
        next_choreo += choreo_len

    train_batches = torch.cat(train_batches, dim=0)
    val_batches = torch.cat(val_batches, dim=0)

    print('######## Data Structures Created For Training ########')

    # Printing all the data structures created
    print('Shape of tensor with all sequences: {}'.format(batches.shape))
    print('Length of each choreography: {}\n'.format(choreo_lens))

    print('Shape of training data with all sequences: {}'.format(train_batches.shape))
    print('Length of each choreography in training dataset: {}\n'.format(train_split))

    print('Shape of validation data with all sequences: {}'.format(val_batches.shape))
    print('Length of each choreography in validation dataset: {}\n'.format(val_split))

    return joint_poses, edge_index_t, batches, choreo_lens, train_batches, train_split, val_batches, val_split