#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/review2/lp_yee_pde_random_01

rm ../lp_output/review2/yee_pde_random_01.txt

./lp -i input/review/input_yee2d_pde_01 -o ../lp_output/review2/lp_yee_pde_random_01 > ../lp_output/review2/yee_pde_random_01.txt
