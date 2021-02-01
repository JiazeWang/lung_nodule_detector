import sys
import UI_util
import numpy as np
import cv2
import time
import math
import torch
import res18_split_focal as detect_model
from torch.nn import DataParallel
from torch.backends import cudnn
import SimpleITK as sitk
from utils import *
from split_combine import SplitComb
#TODO: nodule view rescale feature add
from config import config
class savefile():
    def __init__(self, filename):
        super(savefile,self).__init__()
        self.init_openpath = '/research/dept8/jzwang/code/lung_nodule_detector/dlung_v1/npy/'
        self.resolution = np.array([1,1,1])
        self.slice_index = 0
        self.slice_num = 0
        self.slice_width = 0
        self.slice_height = 0
        self.detect_resume = './detector.ckpt'
        self.gpu = '0'
        self.detect_net, self.split_comber, self.get_pbb \
            = self.init_net()
        self.stride = 4
        self.n_per_run = 1
        self.filename = filename
        self.process(self.filename)


    def init_net(self):
        torch.manual_seed(0)
        torch.cuda.set_device(0)
        detect_config, detect_net, _, get_pbb = detect_model.get_model()
        detect_checkpoint = torch.load(self.detect_resume)
        detect_net.load_state_dict(detect_checkpoint['state_dict'])
        n_gpu = setgpu(self.gpu)
        detect_net = detect_net.cuda()
        cudnn.benchmark = True
        detect_net = DataParallel(detect_net)
        margin = 32
        sidelen = 144
        split_comber = SplitComb(sidelen, detect_config['max_stride'], detect_config['stride'], margin, detect_config['pad_value'])
        return detect_net, split_comber, get_pbb

    def detect(self, filedir, filename):
        self.sliceim_re = np.load(filedir)
        self.slice_arr = np.zeros((np.shape(self.sliceim_re)[0], np.shape(self.sliceim_re)[1], np.shape(self.sliceim_re)[2], 3))
        self.slice_num = np.shape(self.sliceim_re)[0]
        self.slice_height = np.shape(self.sliceim_re)[1]
        self.slice_width = np.shape(self.sliceim_re)[2]
        if (self.slice_num <= 0):
            return 0
        data, coord2, nzhw = UI_util.split_data(np.expand_dims(self.sliceim_re, axis=0),
                                                self.stride, self.split_comber)

        self.world_pbb = UI_util.predict_nodule_v2(self.detect_net, data, coord2, nzhw,
                               self.n_per_run, self.split_comber, self.get_pbb)
        labels_filename = "result/"+filename+".npy"
        np.save(labels_filename, self.world_pbb)

    def process(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        num = 0
        for line in lines:
            print("processing %s"%num)
            num = num + 1
            line = line.rstrip()
            line = "_".join(line.split('/'))
            filedir = self.init_openpath + line + "_clean.npy"
            #print(line)
            self.detect(filedir, line)




if __name__ == '__main__':
    savefile(filename=config["data_txt"])
