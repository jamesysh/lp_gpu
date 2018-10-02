#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

#rm -r ../lp_output/review2/lp_sodshocktube3d_upwind_switch_h_005_oldhex

#rm ../lp_output/review2/sodshocktube3d_upwind_switch_h_1dx_005_oldhex.txt

./lp -i input/review/input_sodshocktube3d_upwind_switch_005 -o ../lp_output/review2/lp_sodshocktube3d_upwind_switch_h_1dx_middis_07_005 > ../lp_output/review2/sodshocktube3d_upwind_switch_h_1dx_middis_07_005.txt
