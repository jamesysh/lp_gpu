#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

#rm -r ../lp_output/lp_nozzle_3d_pipe_lw_sph

#rm ../lp_output/nozzle_3d_pipe_lw_sph.txt

./lp -i input/input_nozzle_3d_pipe_lw_sph -o ../lp_output/lp_nozzle_3d_pipe_lw_sph > ../lp_output/nozzle_3d_pipe_lw_sph.txt
