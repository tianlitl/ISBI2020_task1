#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


This script rearrange the files of datasets in this way:
    
    root/class_x/1.jpg
    root/class_x/2.jpg
    ...
    root/class_y/a.jpg
    root/class_y/b.jpg
    ...

which is the generic data arrangment for torchvision.dataset.DatasetFolder.

In this script, number of class has been set to 5 according to 5 levels of Diabetic Retinopathy.
"""

import csv
import shutil
import re
import os

#####  modify the following values if you need
label_dir = 'F:/DataSet/ISBI2020/ISBI2020_Data/regular-fundus-validation/regular-fundus-validation.csv'
source_root = 'F:/DataSet/ISBI2020/remove_word/sigma10/'
target_dir = 'F:/DataSet/ISBI2020/remove_word/sigma10_arrange/'
#####
if not os.path.exists(target_dir):
    os.mkdir(target_dir[:-1])

for i in range(5):
    if not os.path.exists(target_dir+str(i)):
        os.mkdir(target_dir+str(i))
    
with open(label_dir,'r') as csvfile:
    reader = csv.reader(csvfile)
    reader.__next__()
    img_path_label = [[row[2],row[4],row[5]] for row in reader] 
    # img_label_pair is a list of list in form of
    # [['/regular-fundus-training/1/1_l1.jpg', '0'], ['/regular-fundus-training/1/1_l2.jpg', '0']...]
    for item in img_path_label:
        while '' in item:
            item.remove('')

for item in img_path_label:
    file_name = re.search(r'\d*_.*jpg',item[0]).group() # string in form of '330_l2.jpg' for example
    
    if item[1] == '0':
        shutil.copy(source_root+file_name, target_dir+'0/'+file_name)
    if item[1] == '1':
        shutil.copy(source_root+file_name, target_dir+'1/'+file_name)
    if item[1] == '2':
        shutil.copy(source_root+file_name, target_dir+'2/'+file_name)
    if item[1] == '3':
        shutil.copy(source_root+file_name, target_dir+'3/'+file_name)
    if item[1] == '4':
        shutil.copy(source_root+file_name, target_dir+'4/'+file_name)

    print('{}/{}'.format(item,len(img_path_label)))

