#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/lp_sodshocktube3d_test

rm ../lp_output/sodshocktube3d_test.txt

./lp -i input/review/input_sodshocktube3d_upwind_switch_01 -o ../lp_output/lp_sodshocktube3d_test > ../lp_output/sodshocktube3d_test.txt
