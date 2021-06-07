import json
import numpy as np

filepath = ['/ssd_scratch/cvit/myfolder/idd_data_coco/idd_train_annotation.json' ,'/ssd_scratch/cvit/myfolder/idd_data_coco/idd_val_annotation.json']
#filepath = ['/ssd_scratch/cvit/myfolder/idd_data_coco/idd_val_annotation.json', '/ssd_scratch/cvit/myfolder/idd_data_coco/idd_val_annotation_hold.json']

catg = []
for fpath in filepath:
    with open(fpath) as f:
         gt = json.load(f)

    print(gt['categories'])
    catg.append(gt['categories'])

if catg[0] == catg[1]:
   print("Yes")

print("done")
