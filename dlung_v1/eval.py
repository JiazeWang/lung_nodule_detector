import sys
import UI_util
import numpy as np
import cv2
import time
import math
from xai_viewer_ui import Ui_xai_viewer
import torch
import res18_split_focal as detect_model
import rule
from torch.nn import DataParallel
from torch.backends import cudnn
import SimpleITK as sitk
from utils import *
from split_combine import SplitComb
from save_sitk import savedicom
#TODO: nodule view rescale feature add
class savefile():
    def __init__(self, filename):
        super(savefile,self).__init__()
        self.init_openpath = '/home/imsight/data/demo/'
        self.resolution = np.array([1,1,1])
        self.slice_index = 0
        self.slice_num = 0
        self.slice_width = 0
        self.slice_height = 0
        self.detect_resume = './detector.ckpt'
        self.attribute_resume = './attribute.ckpt'
        self.gpu = '0'
        self.detect_net, self.split_comber, self.get_pbb \
            = self.init_net()
        self.stride = 4
        self.n_per_run = 1
        self.filename = filename
        self.open(self.filename)
        self.detect()


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

        # print ("init_net complete")
        return detect_net, split_comber, get_pbb


    def detect(self):
        if (self.slice_num <= 0):
            return 0
        s = time.time()
        self.gt_path = '/research/dept8/jzwang/code/lung_nodule_integ_viewer/data/' + self.pt_num + '_label.npy'
        data, coord2, nzhw = UI_util.split_data(np.expand_dims(self.sliceim_re, axis=0),
                                                self.stride, self.split_comber)


        labels = np.load(self.gt_path)

        e = time.time()

        self.lbb, self.world_pbb = UI_util.predict_nodule_v1(self.detect_net, data, coord2, nzhw, labels,
                               self.n_per_run, self.split_comber, self.get_pbb)

        nodule_items = []
        for i in range(len(self.lbb)):
            if self.lbb[i][3] != 0:
                nodule_items.append('gt_' + str(i))

        for i in range(len(self.world_pbb)):
            nodule_items.append('cand_' + str(i) + ' ' + str(round(self.world_pbb[i][0], 2)))
        # print('elapsed time is %3.2f seconds' % (e - s))
        UI_util.draw_nodule_rect(self.lbb, self.world_pbb, self.slice_arr)
        num = 0
        # print(" self.slice_arr.shape:", self.slice_arr.shape)
        labels_filename = "result/world_pbb_"+self.pt_num+".npy"
        np.save(labels_filename, self.world_pbb)


    def open(self, file):
        #TODO: file type ch
        fileName = []
        fileName.append(file)
        # print("open ",fileName)
        if (fileName[0] == ''):
            return 0

        self.pt_num = fileName[0].split('/')[-1].split('.mhd')[0]


        sliceim, origin, spacing, isflip = UI_util.load_itk_image(fileName[0])
        # print("sliceim.shape", sliceim.shape)
        # print("origin.shape", origin.shape)
        # print("spacing.shape", spacing.shape)
        # print(spacing)

        if isflip:
            sliceim = sliceim[:, ::-1, ::-1]
            # print('flip!')
        sliceim = UI_util.lumTrans(sliceim)


        self.sliceim_re, _ = UI_util.resample_v1(sliceim, spacing, self.resolution, order=1)
        savedicom(outputdir="result/"+self.pt_num, input=self.sliceim_re,  pixel_dtypes="int16")
        self.slice_arr = np.zeros((np.shape(self.sliceim_re)[0], np.shape(self.sliceim_re)[1], np.shape(self.sliceim_re)[2], 3))
        self.slice_num = np.shape(self.sliceim_re)[0]
        self.slice_height = np.shape(self.sliceim_re)[1]
        self.slice_width = np.shape(self.sliceim_re)[2]
        for i in range(len(self.sliceim_re)):
            self.slice_arr[i] = cv2.cvtColor(self.sliceim_re[i], 8)
        # print ("finish convert")
        self.slice_index = int(self.slice_num/2)
        img = np.array(self.slice_arr[self.slice_index], dtype=np.uint8)





if __name__ == '__main__':
    savefile(filename="/research/dept8/jzwang/code/lung_nodule_integ_viewer/data/001.mhd")
