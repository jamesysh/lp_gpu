#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/lp_simplewave3d_nosplit_fix_01

rm ../lp_output/simplewave3d_nosplit_fix_01.txt

./lp -i input/input_simplewave3d_solidb_lw_nosplit_01 -o ../lp_output/lp_simplewave3d_nosplit_fix_01 > ../lp_output/simplewave3d_nosplit_fix_01.txt
