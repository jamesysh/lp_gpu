#!/bin/bash
#PBS -l nodes=1:ppn=24,walltime=06:00:00
#PBS -N lp_code      
#PBS -q medium-24core
#PBS -o result_o.log
#PBS -e result_e.log

module load shared
module load torque/6.0.2
module load gcc-stack


cd /gpfs/home/nnaitlho/lpcode_032618

./lp -i input/input_pellet_outflow -o /gpfs/projects/SamulyakGroup/Nizar/lp_output_fixed_bc/lp_pellet_outflow > /gpfs/projects/SamulyakGroup/Nizar/lp_output_fixed_bc/pellet_outflow.txt


