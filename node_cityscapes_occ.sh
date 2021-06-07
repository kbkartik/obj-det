#!/bin/bash
#SBATCH -A myfolder
#SBATCH -c 10
#SBATCH --mem-per-cpu=10G
#SBATCH --nodelist=gnode57
#SBATCH --gres=gpu:4
#SBATCH --time=120:00:00

export PATH=$PATH:/bin

cd ~
source activate det_trial

FOLDER="/ssd_scratch/cvit/myfolder/cityscapes"

if [ ! -d "$FOLDER" ]; then
    cd /ssd_scratch/cvit
    mkdir myfolder
    rsync -aqz myfolder@ada:/share3/myfolder/cityscapes /ssd_scratch/cvit/myfolder/
    cd /myfolder/cityped/
    mkdir models
fi

cd ~
#cd detectron2_set2
#rm -rf build/ **/*.so
#python -m pip install -e .
python /home/myfolder/detectron2_set2/tools/cityscapes_train_net.py --num-gpus=4 --dist-url="auto"

