#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

#rm -r ../lp_output/lp_yee3d_pde_fix_01

#rm ../lp_output/yee3d_pde_fix_01.txt

./lp -i input/input_yee3d_pde_01 -o ../lp_output/lp_yee3d_pde_fix_random_01 > ../lp_output/yee3d_pde_fix_random_01.txt
