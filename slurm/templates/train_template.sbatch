#!/bin/bash

# set a job name
{{SET_JOB_NAME}}
# This line sets:
# a file for job output, you can check job progress, append the job ID with %j to make it unique
# a file for errors from the job
{{SET_OUTPUT_PATHS}}

# only use the following if you want email notification
{{SET_EMAIL}}
#SBATCH --mail-type=ALL

#SBATCH --account=juno
#SBATCH --partition=juno --qos=normal
#SBATCH --time=07-00:00:00
#SBATCH --nodes=1

#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

# only use the following on partition with GPUs
# number of GPUS
#SBATCH --gres=gpu:1

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NAME"=$SLURM_JOB_NAME
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

source ~/.bashrc

# Activate conda env
# Change to root directory
{{CONDA_AND_CD_TO_ROOT}}

umask 002

# Launch script
{{CMD}}

# done
echo "Done"
exit 0