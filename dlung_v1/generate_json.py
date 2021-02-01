import numpy as np
import sys

from layers import nms, iou, acc
import time
import multiprocessing as mp
import math
import SimpleITK as sitk
import os
import pandas
import csv
import io
from config import config
import json

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def convert_worldcoord(idx, pbb, filename):
    origin = np.load(filename+'_origin.npy')
    spacing = np.load(filename+'_spacing.npy')
    for label in pbb:
        pos_ori = label[1:4]
        radious_ori = label[4]
        #pos_ori = pos_ori + extendbox[:, 0]
        pos_ori = pos_ori * resolution / spacing
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

def convert_json(input, output, thresholds=0.5):
    with open(input, "r") as f:
        lines = f.readlines()
        lines.append("\n")
    NoduleClass, NoduleScore, NoduleCoordinates, NoduleDiameter= [], [], [], []
    nudule = {}
    num = 0
    result = []
    record = lines[1].split(",")[0]
    for line in lines[1:]:
        nodule_dic = {}
        line = line.rstrip()
        line = line.split(",")
        if line[0] == record and num+1<len(lines[1:]):
            NoduleScore.append(line[-1])
            if float(line[-1])> thresholds:
                NoduleClass.append(1)
            else:
                NoduleClass.append(0)
            NoduleCoordinates.append([line[1], line[2], line[3]])
            NoduleDiameter.append(line[4])
        else:
            nudule = {}
            patient = {"patientName": record, \
                       "nodules": nudule,}
            nudule["NoduleScore"] = NoduleScore
            nudule["NoduleClass"] = NoduleClass
            nudule["NoduleCoordinates"] = NoduleCoordinates
            nudule["NoduleDiameter"] = NoduleDiameter
            NoduleClass, NoduleScore, NoduleCoordinates, NoduleDiameter = [], [], [], []
            NoduleScore.append(line[-1])
            NoduleCoordinates.append([line[1], line[2], line[3]])
            NoduleDiameter.append(line[4])
            record = line[0]
            result.append(patient)
        num = num + 1
    with open(output,'w',encoding='utf-8') as f:
        f.write(json.dumps(result,indent=2))

if __name__ == '__main__':
    pbb = []
    resolution = np.array([1,1,1])
    submit_file = 'submission.txt'
    filename_dict = {}
    csv_submit = []
    csv_sid = []
    with open(config["data_txt"], 'r') as f:
        lines = f.readlines()
    num = 0
    if not os.path.exists(config["result"]):
        os.makedirs(config["result"])
    for i in range(len(lines)):
        print("processing %s"%i)
        line = lines[i].rstrip()
        line = "_".join(line.split('/'))
        pbbdir =  np.load(config["result"] + line + ".npy")
        origin_dir = np.load(config["npy_dir"] + line + "_origin.npy")
        spacing_dir = np.load(config["npy_dir"] + line + "_spacing.npy")
        pbb_item = pbbdir
        filename_dict[i] = config["npy_dir"] + str(line)
        pbb_item = pbb_item[pbb_item[:, 0].argsort()[::-1]]
        pbb_append_list = []
        for item in pbb_item:
            if sigmoid(item[0]) < 0.1:
                continue
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
    pbb = np.array(pbb)
    conf_th = 0.1
    nms_th = 0.3
    detect_th  = 0.3
    for i in range(len(pbb)):
        nms_pbb = nms(pbb[i], nms_th)
        world_pbb = convert_worldcoord(i, nms_pbb, filename_dict[i])
        s_id = filename_dict[i]
        for candidate in world_pbb:
            csv_submit.append([s_id, candidate[1], candidate[2], candidate[3], candidate[4], candidate[0]])

    df_annos = pandas.DataFrame(csv_submit, columns=["seriesuid", "coordX", "coordY", "coordZ", "size", "probability"])
    df_annos.to_csv(submit_file, index=False)
    convert_json('submission.txt', "result.json")
