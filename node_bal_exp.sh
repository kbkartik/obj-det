#!/bin/bash
#SBATCH -A myfolder
#SBATCH -c 4
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:2
#SBATCH --time=180:00:00

export PATH=$PATH:/bin

cd ~
source activate env_det2

FOLDER="/ssd_scratch/cvit/myfolder/balloon"

#rm -rf /ssd_scratch/cvit/myfolder/idd_data_coco

if [ ! -d "$FOLDER" ]; then
    cd /ssd_scratch/cvit
    mkdir myfolder
    rsync -aqz myfolder@ada:/share3/myfolder/balloon /ssd_scratch/cvit/myfolder/
    cd /myfolder/balloon/
    mkdir models
fi


cd ~
cd detectron2_v1 #repo_trial
rm -rf build
cd ~
pip install -e detectron2_v1 #repo_trial
#python /home/myfolder/detectron2_repo_trial/tools/balloon_train_net_experiment.py --num-gpus=2 --dist-url="auto"
python /home/myfolder/detectron2_v1/tools/balloon_train_net.py --num-gpus=2 --dist-url="auto"

