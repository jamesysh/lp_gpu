#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

#rm -r ../lp_output/lp_pellet_test

#rm ../lp_output/pellet_test.txt

./lp -i input/input_pellet_test -o ../lp_output/lp_pellet_v200_large > ../lp_output/pellet_test_v200_large.txt
