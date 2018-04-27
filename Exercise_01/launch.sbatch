#!/usr/bin/env bash
 
#srun --partition=gpu --ntasks=16 --cpus-per-task=1 --mem=170G --qos=short --gres=gpu:8 --pty bash

# This job requests from SLURM to allocate 1 node.
#SBATCH --nodes=1
# On that node, it will run 4 tasks, each with 1 core and 1 GB of memory.
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --qos=medium
#SBATCH --gres=gpu:1

#SBATCH --time=0-20:00:00
# And it will place the output of the commands into my.stdout file
###SBATCH --output=my.stdout
 
# Prepare modules
module load tensorflow/1.5.0-foss-2017a-python-2.7.13-cuda-9.0.176
module load matplotlib/2.0.2-foss-2017a-python-2.7.13
module load pillow/4.3.0-foss-2017a-python-2.7.13

# Executing a command like this will cause it to run once in the whole allocation:
cd /users/manuel.pasieka/projects/HIML/Exercise01
#python GAN02.2.py runtest2 100
python $*
 
# an we will sleep for 60 seconds before exiting
sleep 1
 
# after job is finished, see file my.stdout for results