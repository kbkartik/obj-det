from scipy.io import loadmat
import json

def convert(bbox, img_counter, bbid, istrain):
    
    # Getting well-rounded bbox for class labels: pedestrian and rider
    if bbox[0] == 1 or bbox[0] == 2:
        xmin = bbox[1]
        ymin = bbox[2]
        o_width = bbox[3]
        o_height = bbox[4]

    # Getting approximate-rounded bbox for class labels: pedestrian and rider
    if bbox[0] == 0 or bbox[0] == 3 or bbox[0] == 4 or bbox[0] == 5:
        xmin = bbox[6]
        ymin = bbox[7]
        o_width = bbox[8]
        o_height = bbox[9]

    iscrowd = 0 # Default while training
    ignore = 0 # Default while training

    if istrain == False:
        
        # Ignoring groups/crowd for eval
        if bbox[0] == 5 and bbox[5] < 1000:
            iscrowd = 1
            ignore = 1
            
        #Ignoring 'ignore' class label for eval
        if bbox[0] == 0 and bbox[5] == 0:
            ignore = 1
    
    # Vis_ratio = approximate_bbox_area/well_rounded_bbox_area
    vis_ratio = (bbox[8]*bbox[9])/(bbox[3]*bbox[4])
    category_id = bbox[0]

    annotation = {'area': o_width*o_height, 'iscrowd': iscrowd, 'image_id': img_counter, 'bbox': [xmin, ymin, o_width, o_height], 'height': o_height,
                  'category_id': category_id, 'id': bbid, 'ignore': ignore, 'vis_ratio': vis_ratio, 'segmentation': []}

    return annotation

def main():

    # Pre-defined categories
    categories = [{"id":1,"name":"pedestrian"},{"id":2,"name":"rider"},{"id":3,"name":"sitting person"},{"id":4,"name":"other person"},{"id":5,"name":"people group"},{"id":0,"name":"ignore region"}]

    train_json_dict = {"images":[], "type": "instances", "annotations": [], "categories": categories}
    train_arr = loadmat('/home/kartik/Downloads/anno_train.mat')
    train_arr = train_arr['anno_train_aligned'][0]
    len_train_arr = len(train_arr)
    train_json_file_path = '/home/kartik/Desktop/citypersons/train_gt.json'
    
    eval_json_dict = {"images": [], "annotations": [], "categories": categories}
    eval_arr = loadmat('/home/kartik/Downloads/anno_val.mat')
    eval_arr = eval_arr['anno_val_aligned'][0]
    len_eval_arr = len(eval_arr)
    eval_json_file_path = '/home/kartik/Desktop/citypersons/eval_gt.json'

    bbids = 1
    img_counter = 1
    is_train = True

    # Convert training    
    for idx in range(0, len_train_arr):
        image_annotation_array = train_arr[idx][0][0]
        len_total_bbox_check = len(image_annotation_array[2].tolist())

        # Checking if bboxes exist for this image        
        if len_total_bbox_check > 0:

            bboxes = image_annotation_array[2].tolist()
            image = {'file_name': image_annotation_array[1][0], 'height': 1024, 'width': 2048, 'id': img_counter}
            train_json_dict['images'].append(image)

            for i in range(0, len_total_bbox_check):
                annotation = convert(bboxes[i], img_counter, bbids, is_train)
                train_json_dict['annotations'].append(annotation)
                bbids += 1
            img_counter += 1
    
    bbids = 1
    img_counter = 1
    is_train = False
    
    # Convert evaluation    
    for idx in range(0, len_eval_arr):
        image_annotation_array = eval_arr[idx][0][0]
        len_total_bbox_check = len(image_annotation_array[2].tolist())

        # Checking if bboxes exist for this image        
        if len_total_bbox_check > 0:

            bboxes = image_annotation_array[2].tolist()
            image = {'file_name': image_annotation_array[1][0], 'height': 1024, 'width': 2048, 'id': img_counter}
            eval_json_dict['images'].append(image)

            for i in range(0, len_total_bbox_check):
                annotation = convert(bboxes[i], img_counter, bbids, is_train)
                eval_json_dict['annotations'].append(annotation)
                bbids += 1
            img_counter += 1

    json_fp = open(train_json_file_path, 'w')
    json_str = json.dumps(train_json_dict)
    json_fp.write(json_str)
    json_fp.close()

    json_fp = open(eval_json_file_path, 'w')
    json_str = json.dumps(eval_json_dict)
    json_fp.write(json_str)
    json_fp.close()
    
    print("Done")

if __name__ == '__main__':
  main()
  
"""
        for j in range(0, len_arr):

            cls_lbl = temp_arr[j][0]
            inst_lbl = temp_arr[j][5]

            if class_label == [] and inst_label == []:
                class_label.append(cls_lbl)
                inst_label.append(inst_lbl)
            else:
                if (cls_lbl not in class_label):
                    class_label.append(cls_lbl)
                if (inst_lbl not in inst_label):
                    inst_label.append(inst_lbl)

            count += 1

print("Class label\n")
print(class_label)
print("Instance label\n")
print(inst_label)

print("\n")
print(count)"""
