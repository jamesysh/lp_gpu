#!/bin/bash
#PBS -l nodes=1:ppn=24

cd /home/xingyu/src_08_12_15

rm -r ../lp_output/lp_nozzle_cd

rm ../lp_output/nozzle_cd.txt

./lp -i input/input_nozzle_cd -o ../lp_output/lp_nozzle_cd > ../lp_output/nozzle_cd.txt
