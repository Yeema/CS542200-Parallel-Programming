#!/bin/bash

#SBATCH -N 1 
#SBATCH -n 12
srun ./hybrid_time 4 -2 2 -2 2 1200 1200 out_hybrid.png