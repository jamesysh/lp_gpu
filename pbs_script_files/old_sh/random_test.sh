#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/lp_random_test

rm ../lp_output/random_test.txt

./lp -i input/input_random_test -o ../lp_output/lp_random_test > ../lp_output/random_test.txt