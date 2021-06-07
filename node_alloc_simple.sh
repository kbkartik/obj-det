#!/bin/bash
#SBATCH -A myfolder
#SBATCH -n 3
#SBATCH -c 2
#SBATCH --reservation ndq
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:1
#SBATCH --time=99:30:00

export PATH=$PATH:/bin


source activate env_det2

rsync -aqz myfolder@ada:/share3/myfolder/person_idd_data/val_imgs /ssd_scratch/cvit/myfolder/idd_infer/

cd ~
cd detectron2_v1
rm -rf build
cd ~
pip install -e detectron2_v1
python /home/myfolder/idd_val/visualize_json_results_idd.py --input /home/myfolder/idd_val/coco_instances_results_final.json --output /home/myfolder/idd_val/output_imgs/

