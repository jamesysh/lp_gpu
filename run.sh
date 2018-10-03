#!/bin/bash
#PBS -l nodes=1:ppn=24 
#PBS -q gpu
#PBS -N "test"
module load shared 
module load gcc
module load cuda91
cd ~/lp_gpu/
rm -r output
./lp -i input/input_yee_2d -o output
