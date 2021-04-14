import os
import numpy as np

def list2numpy(list_dir, save_dir):
    pid = []
    with open(list_dir, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        pid.append(line)
    pid = np.array(pid)
    np.save(save_dir, pid)

if __name__ == '__main__':
    list2numpy("hku_list_test.csv", "val_hku.npy")
    list2numpy("hku_list_train.csv", "train_hku.npy")
