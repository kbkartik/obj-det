import json
import numpy as np

file_path = '/home/kartik/Downloads/IDD_Detection/idd_coco_based/all_idd_coco/idd_train_annotation.json'
with open(filepath) as f:
    predictions = json.load(f)

area = []
for p in predictions["annotations"]:
    if p["area"] <= 1024:
        area.append(p["area"])

counts, bins = np.histogram(a, bins=8, range=(0, 1024))

print(counts)
print(bins)
