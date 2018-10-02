#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

#rm -r ../lp_output/lp_random_gaussian

#rm ../lp_output/random_gaussian.txt

./lp -i input/input_random_gaussian -o ../lp_output/lp_random_gaussian > ../lp_output/random_gaussian.txt
