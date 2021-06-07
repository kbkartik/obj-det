import json
import numpy as np

#filepath = ['/ssd_scratch/cvit/myfolder/idd_data_coco/annotations/instances_train2017.json', '/ssd_scratch/cvit/myfolder/idd_data_coco/annotations/instances_val2017.json']
filepath = ['/ssd_scratch/cvit/myfolder/idd_data_coco/idd_train_annotation.json']#, '/ssd_scratch/cvit/myfolder/idd_data_coco/idd_val_annotation.json']
h = 0
w = 0
for fpath in filepath:
    with open(fpath) as f:
         gt = json.load(f)

    print(gt.keys())
    area = []
    max_area = 0
    total_bbox = 0
    for p in gt["annotations"]:
        width = p["bbox"][2]
        height = p["bbox"][3]
        box_area = p["area"]
        if box_area > max_area:
           max_area = box_area
           w = width
           h = height
        area.append(box_area)

        total_bbox += 1
    print("One part complete")
    
    counts, bins = np.histogram(area, bins=[256, 1024, 4096, 16384, 65536, 262144, 1048576, max_area], range=(0, max_area))
    print(counts, bins)
    #print(len(area), total_bbox)
    #print(max_area, w, h)
    
"""
#filepath = ['/ssd_scratch/cvit/myfolder/idd_data_coco/idd_train_annotation.json', '/ssd_scratch/cvit/myfolder/idd_data_coco/idd_val_annotation.json']
filepath = ['/ssd_scratch/cvit/myfolder/idd_data_coco/annotations/instances_train2017.json', '/ssd_scratch/cvit/myfolder/idd_data_coco/annotations/instances_val2017.json']
#newfp = '/ssd_scratch/cvit/myfolder/idd_data_coco/idd_val_annotation_new.json'
list_cats = []
for fpath in filepath:
    with open(fpath) as f:
         gt = json.load(f)
    list_cats.append(gt["categories"])
print(list_cats[0][:10])
print(list_cats[1][:10])
gt["categories"] = list_cats[0]

json_fp = open(newfp, 'w')
json_str = json.dumps(gt)
json_fp.write(json_str)
json_fp.close()"""
print("done")
