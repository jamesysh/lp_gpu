#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/lp_yee_test

rm ../lp_output/yee_test.txt

./lp -i input/input_yee_test -o ../lp_output/lp_yee_test > ../lp_output/yee_test.txt
