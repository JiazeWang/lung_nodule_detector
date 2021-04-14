import numpy as np
import sys

sys.path.append('../')

from training.layers import nms, iou, acc
import time
import multiprocessing as mp
import math
import SimpleITK as sitk
import os
from config_training import config
import pandas
import csv
import io

save_dir = '/research/dept8/jzwang/code/lung_nodule_detector/training/results/test_5_10_30_new/bbox/'
submit_file = './hku_submission_5_new.csv'
sid = './preprocess/hku_list_test.csv'

val_num = np.load('val_hku.npy')
luna_data = '/research/dept8/jzwang/dataset/HKU/TOUSE/'
luna_label = './preprocess/annotations_hku.csv'
resolution = np.array([1,1,1])
annos = np.array(pandas.read_csv(luna_label))



def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def convert_worldcoord(idx, pbb, filename_dict):
    sliceim, origin, spacing = load_itk_image(os.path.join(luna_data, '%s/%s.nii' % (filename_dict[idx], filename_dict[idx])))
    #Mask, extendbox = Mask_info(idx, filename_dict)
    ori_sliceim_shape_yx = sliceim.shape[1:3]
    isflip = 0
    for label in pbb:
        pos_ori = label[1:4]
        radious_ori = label[4]
        #pos_ori = pos_ori + extendbox[:, 0]
        pos_ori = pos_ori * resolution / spacing

        if isflip:
            pos_ori[1:] = ori_sliceim_shape_yx - pos_ori[1:]
            pos_ori[1] = pos_ori[1] * -1
            pos_ori[2] = pos_ori[2] * -1

        pos_ori = pos_ori * spacing
        pos_ori = pos_ori + origin
        pos_ori = pos_ori[::-1]

        radious_ori = radious_ori / spacing[1] * resolution[1]
        radious_ori = radious_ori * spacing[1]

        label[1:4] = pos_ori
        label[4] = radious_ori
        label[0] = sigmoid(label[0])
    return pbb


def duplicate_file(in_filename):
    out_filename = in_filename + '.bin'
    byte_string = ''

    with open(in_filename, 'r') as infile:
        with open(out_filename, 'wb') as outfile:
            char = infile.read(1)
            byte = ord(char)
            # print byte
            byte_string += chr(byte)
            while char != "":
                char = infile.read(1)
                if char != "":
                    byte = ord(char)
                    # print byte
                    byte_string += chr(byte)
            outfile.write(byte_string)
            outfile.close()

if __name__ == '__main__':
    pbb = []
    lbb = []
    filename_dict = {}
    csv_submit = []
    csv_sid = []

    print ("datadir", luna_data)

    for i in range(len(val_num)):
        print("i0:",i)
        pbb_item = np.load(save_dir + str(val_num[i]) + '_pbb.npy')
        lbb_item = np.load(save_dir + str(val_num[i]) + '_lbb.npy')

        filename_dict[i] = str(val_num[i])
        pbb_item = pbb_item[pbb_item[:, 0].argsort()[::-1]]
        pbb_append_list = []
        for item in pbb_item:

            #append nocule prob > 0.1
            if sigmoid(item[0]) < 0.1:
                continue

            #check overlap under 3mm
            is_overlap = False
            for appended in pbb_append_list:
                minimum_dist = 3
                dist = math.sqrt(
                    math.pow(appended[0] - item[0], 2) + math.pow(appended[1] - item[1], 2) + math.pow(
                        appended[2] - item[2], 2))
                if (dist < minimum_dist):
                    is_overlap = True
                    break;

            if not is_overlap:
                pbb_append_list.append(item)

        pbb.append(np.array(pbb_append_list))
        lbb.append(lbb_item)

    pbb = np.array(pbb)
    lbb = np.array(lbb)

    conf_th = 0.1
    nms_th = 0.3
    detect_th  = 0.3

    for i in range(len(pbb)):
        print("i1:",i)
        nms_pbb = nms(pbb[i], nms_th)
        world_pbb = convert_worldcoord(i, nms_pbb, filename_dict)
        print (filename_dict[i])
        s_id = filename_dict[i]
        #csv_sid.append([s_id.encode()])
        csv_sid.append([s_id])
        for candidate in world_pbb:
            csv_submit.append([s_id, candidate[1], candidate[2], candidate[3], candidate[0]])

    df_annos = pandas.DataFrame(csv_submit, columns=["seriesuid", "coordX", "coordY", "coordZ", "probability"])
    df_annos.to_csv(submit_file, index=False)

    df_annos = pandas.DataFrame(csv_sid)
    df_annos.to_csv(sid, index=False, header=False)
