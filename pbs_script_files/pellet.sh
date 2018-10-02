#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/lp_pellet

rm ../lp_output/pellet.txt

./lp -i input/input_pellet -o ../lp_output/lp_pellet > ../lp_output/pellet.txt
