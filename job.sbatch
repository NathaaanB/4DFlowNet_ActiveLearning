#!/bin/bash
#SBATCH --gpus 1
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --mail-type ALL
#SBATCH --output /proj/multipress/users/x_natbl/4DFlowNet/outputfiles/output_train.out
#SBATCH --error /proj/multipress/users/x_natbl/4DFlowNet/outputfiles/output_train.err

cd /proj/multipress/users/x_natbl/4DFlowNet/src
apptainer exec --nv /proj/multipress/users/x_natbl/tensorflow_2.6.1-gpu.sif python trainer.py