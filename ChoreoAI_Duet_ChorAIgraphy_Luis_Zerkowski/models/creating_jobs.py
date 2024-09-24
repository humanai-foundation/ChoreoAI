import os
import subprocess
from itertools import product

# Create a SLURM job script with the specified parameters
def create_slurm_script(script_name, job_name, command, output_file):
    script_content = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time=32:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --qos=shared
#SBATCH --account=<CLUSTER_ACCOUNT>
#SBATCH --output={output_file}

# Loading modules, activating environment and navigating to script directory
module load conda
mamba activate ai_choreo
cd /<PATH_TO_MODELS>/models/

# Running job
srun {command}
'''

    with open(script_name, 'w') as file:
        file.write(script_content)

# Submit a SLURM job script using sbatch
def submit_slurm_job(script_name):
    try:
        subprocess.run(['sbatch', script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e}")

# Generate all combinations of parameters from a dictionary
def generate_param_combinations(params_dict):
    keys, values = zip(*params_dict.items())

    return [dict(zip(keys, v)) for v in product(*values)]

# Tested hyperparameters:
"""
n-sample-dancers = [3, 5],
dataset-multiplier = [10],
compact-encoder = [False, True],
recurrent-encoder = [False],
seq-len-in = [4, 8],
seq-len-out = [1],
hidden-dim = [32, 64],
edge_types = [3, 4],
epochs = [10, 20],
lr = [0.0003], 
lr-cycle = [5],
loss-mode = ['nll', 'mse']
"""

# Hyperparameters to test
param_list = {
    'n-sample-dancers': [3, 5],
    'dataset-multiplier': [10],
    'compact-encoder': [False, True],
    'recurrent-encoder': [False],
    'seq-len-in': [4, 8],
    'seq-len-out': [1],
    'hidden-dim': [32, 64],
    'edge-types': [3, 4],
    'epochs': [20],
    'lr': [0.0003], 
    'lr-cycle': [5],
    'loss-mode': ['nll', 'mse']
}
param_combinations = generate_param_combinations(param_list)

# General variables
script_name = 'train_nri_param_grid.sh'
booleans_list = ['compact-encoder', 'recurrent-encoder']

for d in param_combinations:
    # Using parameters to create different job variables
    job_name = 'param_grid_'
    command = 'python train_nri.py '
    for k, v in d.items():
        job_name += k + '_' + str(v) + '_'
        
        # Treating boolean flags
        if k in booleans_list:
            if v is False:
                continue
            else:
                command += '--' + k + ' '

        else:
            command += '--' + k + ' ' + str(v) + ' '
    
    job_name = job_name[:-1]
    command += '--save-path best_weights/nri_parameters_' + job_name + '.pt '
    command += '--visual assets/' + job_name + '.png'

    output_file = job_name + '.out'

    create_slurm_script(script_name, job_name, command, output_file)
    submit_slurm_job(script_name)