import lxml.etree as ET
from shutil import copyfile
import time

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

base_path = "/home/kartik/Downloads/idd-detection/IDD_Detection/"
orig_ann_path = base_path + "Annotations/"                          # Original path of all annotation xmls
orig_img_path = base_path + "JPEGImages/"                           # Original path of all image files
train_filepath = base_path + "train.txt"                            # Original text file containing paths of training/validation files
train_xmllist = base_path + "all_train_xml_list.txt"                # New text file containing path lists of xml files
out_train_ann_path = base_path + "all_train_ann/"                   # New folder consisting of renamed training/validation annotation xmls
out_train_img_path = base_path + "all_train_imgs/"                  # New folder consisting of renamed training/validation images

file = open(train_filepath)
lines = [line.rstrip('\n') for line in file]
file.close()

train_xmls_list = []

i = 1

#lines = lines[:1]

start_time = time.time()

for file_line in lines:
    main_tree = ET.parse(orig_ann_path + file_line + ".xml")
    root = main_tree.getroot()
    root_children = root.getchildren()
    #obj_children = root_children[3:] # alternative: main_tree.xpath('./object[./name="person"]')
                                      #  --> using xpath to get 'object' elements containing person as label

    new_root = ET.Element("annotation")
    filename = ET.SubElement(new_root, "filename")
    filename.text = str(i)+".jpg"
    folder = ET.SubElement(new_root, "folder")
    folder.text = "all_train_imgs"
    root_children[0].tag = "orig_filename"
    root_children[1].tag = "orig_folder"

    # Appending person-based element objects to xml tree
    for get_obj in root_children:
        new_root.append(get_obj)
        
    # Indenting and writing new xml files
    indent(new_root)
    ET.ElementTree(new_root).write((out_train_ann_path+str(i)+".xml"), encoding='utf-8', xml_declaration=True)

    # appending person-based xml annotation files to list
    train_xmls_list.append(str(i)+".xml")

    # Copy person-based img files into one folder
    copyfile(orig_img_path + file_line + ".jpg", out_train_img_path + str(i) + ".jpg")
   
    i += 1

# Creating xml list of all training/validation annotations
file = open(train_xmllist, 'w')
for xmlfile in train_xmls_list:
    file.write(xmlfile+"\n")
file.close()

print("--- %s seconds ---" % (time.time() - start_time))

print("Success")
