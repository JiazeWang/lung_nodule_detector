import os

# we need to change base only.Then put the data under BASE/data/sample
BASE = os.path.realpath(__file__)[:-9]  # assume the BASE dir is 'data'  ' # make sure we have the ending '/'
data_config = {
    "img_dir" : BASE + "data/",
    "data_txt" : "hkt_list.txt",
    "lung_mask_dir" : "/research/dept8/jzwang/dataset/HKU/preprocessed/lung_mask/",
    "npy_dir" :  BASE +  "data/hku_npy/",
    "mhd_dir": BASE + "data/mhd/",
    "result": BASE + "result/",
    "record": "record_series_list.txt",
}
config = dict(data_config)
