#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

#rm -r ../lp_output/review2/lp_sodshocktube3d_upwind_switch_0005

#rm ../lp_output/review2/sodshocktube3d_upwind_switch_0005.txt

./lp -i input/review/input_sodshocktube3d_upwind_switch_0005 -o ../lp_output/review2/lp_sodshocktube3d_upwind_switch_h_4dx_0005_fit > ../lp_output/review2/sodshocktube3d_upwind_switch_h_4dx_0005_fit.txt
