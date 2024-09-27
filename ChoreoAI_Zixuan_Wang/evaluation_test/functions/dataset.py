import torch
from functions.load_data import * 
from torch_geometric.data import Data

class MarielDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, reduced_joints=False, xy_centering=True, seq_len=128, file_path="data/mariel_*.npy", no_overlap=False):
        'Initialization'
        self.file_path      = file_path
        self.seq_len        = seq_len
        self.no_overlap     = no_overlap
        self.reduced_joints = reduced_joints # use a meaningful subset of joints
        self.data           = load_data(pattern=file_path) 
        self.xy_centering   = xy_centering
        self.n_joints       = 53
        self.n_dim          = 6
        
        print("")
        
        if self.no_overlap:
            print("Generating non-overlapping sequences...")   
        else:
            print("Generating overlapping sequences...")
        
        if self.xy_centering:
            print("Using (x, y)-centering...")
        else:
            print("Not using (x, y)-centering...")
            
        if self.reduced_joints: 
            print("Reducing joints...")
        else:
            print("Using all joints...")

    def __len__(self):
        'Denotes the total number of samples'
        if self.xy_centering: 
            data = self.data[1] # choose index 1, for the (x,y)-centered phrases
        else: 
            data = self.data[0] # choose index 0, for data without (x,y)-centering
        
        if self.no_overlap:
             # number of complete non-overlapping phrases
            return int(len(data) / self.seq_len)
        else:
            # number of overlapping phrases up until the final complete phrase
            return len(data) - self.seq_len 

    def __getitem__(self, index):
        'Generates one sample of data'
        edge_index, is_skeleton_edge, reduced_joint_indices = edges(reduced_joints=self.reduced_joints, seq_len=self.seq_len)
        
        if self.xy_centering:
            data = self.data[1] # choose index 1, for the (x,y)-centered phrases
        else:
            data = self.data[0] # choose index 0, for data without (x,y)-centering

        if self.reduced_joints:
            data = data[:, reduced_joint_indices, :] # reduce number of joints if desired

        if self.no_overlap:
            # non-overlapping phrases
            index = index * self.seq_len
            sequence = data[index: index + self.seq_len]
            # prediction_target = data[index: index + self.seq_len + self.predicted_timesteps]
        else:
            # overlapping phrases
            sequence = data[index: index + self.seq_len]
            # prediction_target = data[index: index + self.seq_len + self.predicted_timesteps]

        # [seq_len, joints, 6]
        cur_data = {}
        cur_data['seq'] = torch.Tensor(sequence)
        # cur_data['target'] = torch.Tensor(prediction_target)
        return cur_data
    