import lxml.etree as ET
from shutil import copyfile

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
orig_ann_path = base_path + "Annotations/"                              # Original path of all annotation xmls
orig_img_path = base_path + "JPEGImages/"                               # Original path of all image files
val_filepath = base_path + "val.txt"                                # Original text file containing paths of training/validation files
non_person_val_xml_filepath = base_path + "non_person_val.txt"      # New text file containing paths having no persons in images
person_val_xml_filepath = base_path + "person_val.txt"              # New text file containing paths having persons in images
val_xmllist = base_path + "val_xml_list.txt"                        # New text file containing path lists of xml files
out_val_ann_path = base_path + "val_ann/"                           # New folder consisting of training/validation annotation xmls which have atleast one person object
out_val_img_path = base_path + "val_imgs/"                          # New folder consisting of training/validation images which have atleast one person object

file = open(val_filepath)
lines = [line.rstrip('\n') for line in file]
file.close()

non_person_val_xmls = []
person_val_xmls = []
val_xmls_list = []

i = 1

#lines = lines[:10]

for file_line in lines:
    main_tree = ET.parse(orig_ann_path + file_line + ".xml")
    if (main_tree.xpath('count(/annotation/object/name[.="person"])')) >= 1:
        root = main_tree.getroot()
        root_children = root.getchildren()
        obj_children = root_children[3:] # alternative: main_tree.xpath('./object[./name="person"]')
                                         #  --> using xpath to get 'object' elements containing person as label

        new_root = ET.Element("annotation")
        filename = ET.SubElement(new_root, "filename")
        filename.text = str(i)+".jpg"
        folder = ET.SubElement(new_root, "folder")
        folder.text = "val"
        root_children[0].tag = "orig_filename"
        root_children[1].tag = "orig_folder"
        new_root.append(root_children[0])
        new_root.append(root_children[1])
        new_root.append(root_children[2])

        # Appending person-based element objects to xml tree
        for get_person_obj in obj_children:
            if (get_person_obj.find('name')).text == 'person':
                new_root.append(get_person_obj)

        # Indenting and writing new xml files
        indent(new_root)
        ET.ElementTree(new_root).write((out_val_ann_path+str(i)+".xml"), encoding='utf-8', xml_declaration=True)

        # appending person-based xml annotation files to list
        val_xmls_list.append(str(i)+".xml")

        # Copy person-based img files into one folder
        copyfile(orig_img_path + file_line + ".jpg", out_val_img_path + str(i) + ".jpg")

        person_val_xmls.append(file_line)

        i += 1

    else:
        non_person_val_xmls.append(file_line)

# Creating xml list of all training/validation annotations
file = open(val_xmllist, 'w')
for xmlfile in val_xmls_list:
    file.write(xmlfile+"\n")
file.close()

# Creating non-person original xml filepaths
file = open(non_person_val_xml_filepath, 'w')
for xmlfile in non_person_val_xmls:
    file.write(xmlfile+"\n")
file.close()

# Creating person original xml filepaths
file = open(person_val_xml_filepath, 'w')
for xmlfile in person_val_xmls:
    file.write(xmlfile+"\n")
file.close()

print("Success")
