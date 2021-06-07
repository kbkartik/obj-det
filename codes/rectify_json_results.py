import json
#from fvcore.common.file_io import PathManager

#with PathManager.open(args.input, "r") as f:
#    predictions = json.load(f)

file_path = '/home/kartik/Desktop/coco_instances_results_final.json'
with open('/home/kartik/Desktop/coco_instances_results_person.json') as f:
    predictions = json.load(f)

i = 1
m = 0
for p in predictions["annotations"]:
    predictions["annotations"][m]["id"] = i
    i += 1
    m += 1

json_fp = open(file_path, 'w')
json_str = json.dumps(predictions)
json_fp.write(json_str)
json_fp.close()
