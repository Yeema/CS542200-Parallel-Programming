#!/bin/bash

#SBATCH -N 1 
#SBATCH -n 12
srun ./mpi_dynamic_time 4 -2 2 -2 2 4800 4800 out_mpi_dynamic.png