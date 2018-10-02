#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/lp_random_multigaussian

rm ../lp_output/random_multigaussian.txt

./lp -i input/input_random_multigaussian -o ../lp_output/lp_random_multigaussian > ../lp_output/random_multigaussian.txt
