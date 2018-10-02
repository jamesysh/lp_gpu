#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/lp_pellet_outflow_0005

rm ../lp_output/pellet_outflow_0005.txt

./lp -i input/input_pellet_outflow_0005 -o ../lp_output/lp_pellet_outflow_0005 > ../lp_output/pellet_outflow_0005.txt
