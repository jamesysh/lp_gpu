#!/bin/bash
#PBS -l nodes=1:ppn=20

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/review2/lp_simplewave_nosplit_review_005

rm ../lp_output/review2/simplewave_nosplit_review_005.txt

./lp -i input/input_simplewave2d_solidb_lw_nosplit_005 -o ../lp_output/review2/lp_simplewave_nosplit_review_005 > ../lp_output/review2/simplewave_nosplit_review_005.txt

