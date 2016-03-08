import os
import numpy as np
import h5py
import cPickle
import sys

sys.setrecursionlimit(50000)

# loading and preparint the SVHN dataset
dataset_path = "/CK/SVHN"
train_data_path = os.path.join(dataset_path,"train")
test_data_path = os.path.join(dataset_path,"test")

mat_file_name = "digitStruct.mat"    # contains the groundtruth information
file_ext = lambda x: x.endswith('.png') # the images are in png format

# read the .mat file in train and test data
print " Reading the train data info from the .mat file..."
train_digit_struct = {}
with h5py.File(os.path.join(train_data_path,mat_file_name),'r') as train_mat_file:
    bboxes = train_mat_file['digitStruct/bbox']
    names = train_mat_file['digitStruct/name']
    for index in xrange(len(names)):
        name_hdf5 = names[index][0]
        name_obj = train_mat_file[name_hdf5]
        name = ''.join(chr(i) for i in name_obj[:])
        train_digit_struct[name] = {}
    
        bbox_hdf5 = bboxes[index][0]
        bbox_obj = train_mat_file[bbox_hdf5]
        for key in bbox_obj.keys():
            key = str(key)
            for index2 in range(len(bbox_obj[key])):
                try:
                    train_digit_struct[name][key].append(int(train_mat_file[bbox_obj[key][index2][0]][0][0]))
                except Exception as e:
                    if len(bbox_obj[key])>1:
                        train_digit_struct[name][key] = [int(train_mat_file[bbox_obj[key][index2][0]][0][0])]
                    else:
                        train_digit_struct[name][key] = [int(bbox_obj[key][index2][0])]

print " Writing train data structure"
f = file('train_digit_struct.save', 'wb')
cPickle.dump(train_digit_struct, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

print " Reading the test data info from the .mat file..."
test_digit_struct = {}
with h5py.File(os.path.join(test_data_path,mat_file_name),'r') as test_mat_file:
    bboxes = test_mat_file['digitStruct/bbox']
    names = test_mat_file['digitStruct/name']

    for index in xrange(len(names)):
        name_hdf5 = names[index][0]
        name_obj = test_mat_file[name_hdf5]
        name = ''.join(chr(i) for i in name_obj[:])
        test_digit_struct[name] = {}
    
        bbox_hdf5 = bboxes[index][0]
        bbox_obj = test_mat_file[bbox_hdf5]
        for key in bbox_obj.keys():
            key = str(key)
            for index2 in range(len(bbox_obj[key])):
                try:
                    test_digit_struct[name][key].append(int(test_mat_file[bbox_obj[key][index2][0]][0][0]))
                except Exception as e:
                    if len(bbox_obj[key])>1:
                        test_digit_struct[name][key] = [int(test_mat_file[bbox_obj[key][index2][0]][0][0])]
                    else:
                        test_digit_struct[name][key] = [int(bbox_obj[key][index2][0])]

print " Writing test data structure"
f = file('test_digit_struct.save', 'wb')
cPickle.dump(test_digit_struct, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
