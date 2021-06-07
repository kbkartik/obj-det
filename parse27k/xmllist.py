# Import packages
from os import listdir

train_path = '/home/kartik/Desktop/parse27k_ready/my_parse/annotation/train/'
val_path = '/home/kartik/Desktop/parse27k_ready/my_parse/annotation/val/'
train_xml_list = '/home/kartik/Desktop/parse27k_ready/train_xmllist_parse27k.txt'
val_xml_list = '/home/kartik/Desktop/parse27k_ready/val_xmllist_parse27k.txt'

i = 0
file = open(train_xml_list, 'w')
for file_name in listdir(train_path):
	file.write(file_name+"\n")
	i += 1
file.close()

print(i)

i = 0
file = open(val_xml_list, 'w')
for file_name in listdir(val_path):
	file.write(file_name+"\n")
	i += 1
file.close()

print(i)

