#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/review2/lp_sodshocktube_equalmass_upwind_switch_002

rm ../lp_output/review2/sodshocktube_equalmass_upwind_switch_002.txt

./lp -i input/review/input_sodshocktube_equalmass_upwind_switch_002 -o ../lp_output/review2/lp_sodshocktube_equalmass_upwind_switch_002 > ../lp_output/review2/sodshocktube_equalmass_upwind_switch_002.txt
