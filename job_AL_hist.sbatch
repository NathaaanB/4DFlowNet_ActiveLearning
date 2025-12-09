#!/bin/bash
#SBATCH --gpus 1
#SBATCH -N 1
#SBATCH -t 70:00:00
#SBATCH --mail-type ALL
#SBATCH --output /proj/multipress/users/x_natbl/4DFlowNet_AL/outputfiles/output_train_al_rs.out
#SBATCH --error /proj/multipress/users/x_natbl/4DFlowNet_AL/outputfiles/output_train_al_rs.err

cd /proj/multipress/users/x_natbl/4DFlowNet_AL/src
apptainer exec --nv /proj/multipress/users/x_natbl/tensorflow_2.6.1-gpu.sif python -m metrics.error_vs_unc_patches.patch_traj