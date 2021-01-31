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
class savefile():
    def __init__(self, filename):
        super(savefile,self).__init__()
        self.init_openpath = '/research/dept8/jzwang/dataset/LUNA16/combined/'
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

        #model = import_module(self.model)
        detect_config, detect_net, _, get_pbb = detect_model.get_model()

        detect_checkpoint = torch.load(self.detect_resume)
        detect_net.load_state_dict(detect_checkpoint['state_dict'])


        n_gpu = setgpu(self.gpu)

        detect_net = detect_net.cuda()
        #loss = loss.cuda()
        cudnn.benchmark = True
        detect_net = DataParallel(detect_net)

        margin = 32
        sidelen = 144
        split_comber = SplitComb(sidelen, detect_config['max_stride'], detect_config['stride'], margin, detect_config['pad_value'])
        return detect_net, split_comber, get_pbb


    def detect(self):
        if (self.slice_num <= 0):
            return 0
        data, coord2, nzhw = UI_util.split_data(np.expand_dims(self.sliceim_re, axis=0),
                                                self.stride, self.split_comber)

        self.world_pbb = UI_util.predict_nodule_v2(self.detect_net, data, coord2, nzhw,
                               self.n_per_run, self.split_comber, self.get_pbb)
        labels_filename = "result/"+self.pt_num+".npy"
        np.save(labels_filename, self.world_pbb)


    def openfile(self, filename):
        #TODO: file type

        self.pt_num = filename.split('/')[-1].split('.mhd')[0]
        sliceim, origin, spacing, isflip = UI_util.load_itk_image(filename)
        if isflip:
            sliceim = sliceim[:, ::-1, ::-1]
            # print('flip!')
        sliceim = UI_util.lumTrans(sliceim)


        self.sliceim_re, _ = UI_util.resample_v1(sliceim, spacing, self.resolution, order=1)
        self.slice_arr = np.zeros((np.shape(self.sliceim_re)[0], np.shape(self.sliceim_re)[1], np.shape(self.sliceim_re)[2], 3))
        self.slice_num = np.shape(self.sliceim_re)[0]
        self.slice_height = np.shape(self.sliceim_re)[1]
        self.slice_width = np.shape(self.sliceim_re)[2]
        for i in range(len(self.sliceim_re)):
            self.slice_arr[i] = cv2.cvtColor(self.sliceim_re[i], 8)
        # print ("finish convert")
        self.slice_index = int(self.slice_num/2)
        img = np.array(self.slice_arr[self.slice_index], dtype=np.uint8)

    def process(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        num = 0
        for line in lines:
            print("processing %s"%num)
            num = num + 1
            line = line.rstrip()
            line = self.init_openpath + line
            print(line)
            self.openfile(line)
            self.detect()





if __name__ == '__main__':
    savefile(filename="/research/dept8/jzwang/dataset/LUNA16/list.txt")
