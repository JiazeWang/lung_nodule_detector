import os

BASE = '/research/dept8/jzwang/code/lung_nodule_detector/dlung_v1/' # make sure you have the ending '/'
data_config = {
    "img_dir" : BASE + "data/",
    "data_txt" : BASE + "data/filedir.txt",
    "lung_mask_dir" : BASE + "data/lung_mask/",
    "npy_dir" : "data/npy/",
    "result": BASE + "result/"
}
config = dict(data_config)