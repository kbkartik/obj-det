import lxml.etree as ET

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

inp_path = "/home/kartik/Desktop/000084_r.xml"
xml_out = "/home/kartik/Desktop/Ann/2.xml"

i = 2

main_tree = ET.parse(inp_path)
#print(main_tree.xpath('count(/annotation/object/name[.="person"])'))
#print(main_tree.xpath('./object[./name="person"]'))

root = main_tree.getroot()
root_children = root.getchildren()
obj_children = root_children[3:]

new_root = ET.Element("annotation")
filename = ET.SubElement(new_root, "filename")
filename.text = str(i)+".jpg"
folder = ET.SubElement(new_root, "folder")
folder.text = "train"
root_children[0].tag = "orig_filename"
root_children[1].tag = "orig_folder"
new_root.append(root_children[0])
new_root.append(root_children[1])
new_root.append(root_children[2])

for get_person_obj in obj_children:
    if (get_person_obj.find('name')).text == 'person':
        new_root.append(get_person_obj)

indent(new_root)
ET.ElementTree(new_root).write(xml_out)
