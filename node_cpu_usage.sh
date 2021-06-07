#!/bin/bash
#SBATCH -A myfolder
#SBATCH -c 5
#SBATCH --mem-per-cpu=5G
#SBATCH --nodelist=gnode61
#SBATCH --time=180:00:00

export PATH=$PATH:/bin

cd ~
#source activate env_det2

python /home/myfolder/check_annotation_json.py  #area_small_idd_statistics.py
