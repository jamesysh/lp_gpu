#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/jet3d_test

rm ../lp_output/jet3d_test.txt

./lp -i input/input_jet3d_test -o ../lp_output/jet3d_test > ../lp_output/jet3d_test.txt
