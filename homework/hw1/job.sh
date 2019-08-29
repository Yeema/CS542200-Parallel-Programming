#!/bin/bash
#SBATCH -n 12
#SBATCH -N 1
srun ./advanced 536869888 /home/ta/hw1/testcases/33.in 33.out
srun ./advanced 536869888 /home/ta/hw1/testcases/34.in 34.out
