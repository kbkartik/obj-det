#!/bin/bash
#SBATCH -A myfolder
#SBATCH -c 10
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gnode60
#SBATCH --time=150:00:00

export PATH=$PATH:/bin

cd ~
source activate env_det2

FOLDER="/ssd_scratch/cvit/myfolder/idd_data_coco"

if [ ! -d "$FOLDER" ]; then
    cd /ssd_scratch/cvit/myfolder
    #mkdir myfolder
    rsync -aqz myfolder@ada:/share3/myfolder/idd_data_coco /ssd_scratch/cvit/myfolder/
    cd /ssd_scratch/cvit/myfolder/idd_data_coco/
    mkdir models
fi


cd ~
#cd detectron2_set1
#rm -rf build/ **/*.so
#python -m pip install -e .
python /home/myfolder/detectron2_set1/tools/idd_train_net.py --num-gpus=4 --dist-url="auto"
