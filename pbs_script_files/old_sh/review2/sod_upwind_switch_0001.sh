#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

#rm -r ../lp_output/review2/lp_sodshocktube_upwind_switch_0001

#rm ../lp_output/review2/sodshocktube_upwind_switch_0001.txt

./lp -i input/review/input_sodshocktube_upwind_switch_0001 -o ../lp_output/review2/lp_sodshocktube_upwind_switch_0001 > ../lp_output/review2/sodshocktube_upwind_switch_0001.txt
