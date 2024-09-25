#!/bin/bash
#SBATCH --job-name=param_grid_n-sample-dancers_5_dataset-multiplier_10_compact-encoder_True_recurrent-encoder_False_seq-len-in_8_seq-len-out_1_hidden-dim_64_edge-types_4_epochs_20_lr_0.0003_lr-cycle_5_loss-mode_mse
#SBATCH --time=32:00:00
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --qos=shared
#SBATCH --account=<CLUSTER_ACCOUNT>
#SBATCH --output=param_grid_n-sample-dancers_5_dataset-multiplier_10_compact-encoder_True_recurrent-encoder_False_seq-len-in_8_seq-len-out_1_hidden-dim_64_edge-types_4_epochs_20_lr_0.0003_lr-cycle_5_loss-mode_mse.out

# Loading modules, activating environment and navigating to script directory
module load conda
mamba activate ai_choreo
cd /<PATH_TO_MODELS>/models/

# Running job
srun python train_nri.py --n-sample-dancers 5 --dataset-multiplier 10 --compact-encoder --seq-len-in 8 --seq-len-out 1 --hidden-dim 64 --edge-types 4 --epochs 20 --lr 0.0003 --lr-cycle 5 --loss-mode mse --save-path best_weights/nri_parameters_param_grid_n-sample-dancers_5_dataset-multiplier_10_compact-encoder_True_recurrent-encoder_False_seq-len-in_8_seq-len-out_1_hidden-dim_64_edge-types_4_epochs_20_lr_0.0003_lr-cycle_5_loss-mode_mse.pt --visual assets/param_grid_n-sample-dancers_5_dataset-multiplier_10_compact-encoder_True_recurrent-encoder_False_seq-len-in_8_seq-len-out_1_hidden-dim_64_edge-types_4_epochs_20_lr_0.0003_lr-cycle_5_loss-mode_mse.png
