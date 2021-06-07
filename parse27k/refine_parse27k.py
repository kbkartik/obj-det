# Import packages
import sqlite3 as lite
import lxml.etree as ET
from shutil import copyfile
from os import listdir
import matplotlib.image as mpimg

# insert customized value into dictionary
def insert_train_val_dict(image_id, bbox_ids, bbox_ids_length, path, file_name, pedestrian_table_dict, AttSet_table_dict):
    
    bbox_gender = []

    for itr in range(0, bbox_ids_length):
        bbox = pedestrian_table_dict[bbox_ids[itr]]
        gender = AttSet_table_dict[bbox_ids[itr][1]]
        bbox.append(gender)
        bbox_gender.append(bbox)

    return [image_id, [640, 480], bbox_gender]

# Getting training, validation into a single dict (Same for testing, if any file exists)
def get_train_val_test_dicts(image_table_dict, pedestrian_table_dict, AttSet_table_dict, Sequence_table_dict):

    train_val_count = 0
    test_count = 0

    train_val_dict = {}
    test_dict = {}

    for seqID in range(1,9):

        img_directory = Sequence_table_dict[seqID]
        path = '/home/kartik/Desktop/datasets/parse27k/sequences/' + img_directory + '/'

        for file_name in listdir(path):

            image_id = image_table_dict[(file_name, seqID)]
            bbox_ids = sorted([ k for k in pedestrian_table_dict.keys() if k[0] == image_id ])
            bbox_ids_length = len(bbox_ids)
            dict_key = img_directory + '/' + file_name
            
            if bbox_ids_length > 0:

                if train_val_dict == {} or (dict_key not in train_val_dict.keys()):
                    train_val_dict[dict_key] = insert_train_val_dict(image_id, bbox_ids,
                                                                     bbox_ids_length, path, file_name,
                                                                     pedestrian_table_dict, AttSet_table_dict)

                    train_val_count += 1
                else:
                    print("Train Val Dict SeqID {} img_directory {} file_name {}".format(seqID, img_directory, file_name))
                
            elif bbox_ids_length == 0:

                if test_dict == {} or (dict_key not in test_dict.keys()):
                    test[dict_key] = dict_key
                    test_count += 1
                else:
                    print("Test Dict SeqID {} img_directory {} file_name {}".format(seqID, img_directory, file_name))

    return train_val_dict, test_dict, train_val_count, test_count

# indenting xml annotation file as per standard format
def indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


# Generate xmls
def generatexmls(train_val_dict, train_val_count):

    annotation_path = '/home/kartik/Desktop/datasets/my_parse/annotation/'
    image_path = '/home/kartik/Desktop/datasets/my_parse/images/'
    original_image_path = '/home/kartik/Desktop/datasets/parse27k/sequences/'

    train_val_keys = list(train_val_dict.keys())
    train_count = int(0.6*train_val_count)
    val_count = int(0.8*train_val_count)
        
    for i in range(0, train_val_count):

        key = train_val_keys[i]

        new_root = ET.Element("annotation")
        filename = ET.SubElement(new_root, "filename")
        filename.text = str(train_val_dict[key][0])+".png"
        folder = ET.SubElement(new_root, "folder")
            
        if i >= 0 and i <= train_count:
            folder.text = "train"
        elif i >= (train_count+1) and i <= val_count:
            folder.text = "val"
        elif i >=  (val_count+1) and i <= train_val_count:
            folder.text = "test"

        temp_key = key.split('/')

        orig_filename = ET.SubElement(new_root, "orig_filename")
        orig_filename.text = temp_key[1]
        orig_folder = ET.SubElement(new_root, "orig_folder")
        orig_folder.text = temp_key[0]

        size = ET.SubElement(new_root, "size")
        width = ET.SubElement(size, "width")
        width.text = str(train_val_dict[key][1][0])
        height = ET.SubElement(size, "height")
        height.text = str(train_val_dict[key][1][1])
        depth = ET.SubElement(size, "depth")
        depth.text = str(3)

        total_bndbox_img = len(train_val_dict[key][2])

        for m in range(0, total_bndbox_img):

            bbox = train_val_dict[key][2][m]
            
            bbox_obj = ET.SubElement(new_root, "object")
            name = ET.SubElement(bbox_obj, "name")
            name.text = "person"
            bndbox = ET.SubElement(bbox_obj, "bndbox")

            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(bbox[0])
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(bbox[1])
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(bbox[2])
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(bbox[3])
      
        # Indenting and writing new xml files
        indent(new_root)

        if i >= 0 and i <= train_count:
            ET.ElementTree(new_root).write(annotation_path+'train/'+str(train_val_dict[key][0])+".xml", encoding='utf-8', xml_declaration=True)
            copyfile(original_image_path + key, image_path+'train/'+str(train_val_dict[key][0])+".png")
            
        elif i >= (train_count+1) and i <= val_count:
            ET.ElementTree(new_root).write(annotation_path+'val/'+str(train_val_dict[key][0])+".xml", encoding='utf-8', xml_declaration=True)
            copyfile(original_image_path + key, image_path+'val/'+str(train_val_dict[key][0])+".png")
        elif i >= (val_count+1) and i <= train_val_count:
            ET.ElementTree(new_root).write(annotation_path+'test/'+str(train_val_dict[key][0])+".xml", encoding='utf-8', xml_declaration=True)
            copyfile(original_image_path + key, image_path+'test/'+str(train_val_dict[key][0])+".png")


print("Main\n")
    
image_table_dict = {}
pedestrian_table_dict = {}
AttSet_table_dict = {}
Sequence_table_dict = {}

conn = lite.connect('/home/kartik/Desktop/datasets/parse27k/annotations.sqlite3')
conn.row_factory = lite.Row
cur = conn.cursor()

# Mapping filename and sequenceID to imageID in a dictionary
print("Table Image")
cur.execute("SELECT * FROM 'Image'")
for row in cur:
    dict_key = (row["filename"],row["sequenceID"])
    
    if image_table_dict == {} or (dict_key not in image_table_dict.keys()):
        image_table_dict[dict_key] = row["imageID"]
    else:
        print(dict_key)


# Mapping imageID, attributeSetID to bounding boxes in a dictionary
print("Table Pedestrian")
cur.execute("SELECT * FROM 'Pedestrian'")
for row in cur:
    dict_key = (row["imageID"], row["attributeSetID"])
    
    if pedestrian_table_dict == {} or (dict_key not in pedestrian_table_dict.keys()):
        pedestrian_table_dict[(row["imageID"], row["attributeSetID"])] = [row["box_min_x"], row["box_min_y"], row["box_max_x"], row["box_max_y"]]
    else:
        print(dict_key)

# Mapping attributeSetID to gender in a dictionary
print("Table AttributeSet")
cur.execute("SELECT * FROM 'AttributeSet'")
for row in cur:
    dict_key = row["attributeSetID"]

    if AttSet_table_dict == {} or (dict_key not in AttSet_table_dict.keys()):
        if row["genderID"] == 2:
            AttSet_table_dict[dict_key] = "Male"
        elif row["genderID"] == 3:
            AttSet_table_dict[dict_key] = "Female"
        elif row["genderID"] == 1:
            AttSet_table_dict[dict_key] = "NA"
    else:
        print(dict_key)

# Mapping sequenceId to directory name in a dictionary
print("Table Sequence")
cur.execute("SELECT * FROM 'Sequence'")
for row in cur:
    dict_key = row["sequenceID"]

    if Sequence_table_dict == {} or (dict_key not in Sequence_table_dict.keys()):
        Sequence_table_dict[dict_key] = row["directory"]
    else:
        print(dict_key)
            
conn.close()

train_val_dict, test_dict, train_val_count, test_count = get_train_val_test_dicts(image_table_dict, pedestrian_table_dict, AttSet_table_dict, Sequence_table_dict)

if test_count == 0 and test_dict == {}:
    print("No test case")

generatexmls(train_val_dict, train_val_count)

print("Success")
