#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/lp_yee_2d

rm ../lp_output/yee_2d.txt

./lp -i input/input_yee_2d -o ../lp_output/lp_yee_2d > ../lp_output/yee_2d.txt
