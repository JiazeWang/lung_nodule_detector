import os
import shutil
import numpy as np
from scipy.ndimage.interpolation import zoom
import SimpleITK as sitk
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
import pandas
import sys
import math
import glob
from bs4 import BeautifulSoup
from multiprocessing import Pool

def get_lung(filename, output):
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(filename)
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    segmentation = mask.apply(img)
    result_out= sitk.GetImageFromArray(segmentation)
    output = output+'.mhd'
    sitk.WriteImage(result_out, output)

def resample(imgs, spacing, new_spacing,order=2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        imgs = zoom(imgs, resize_factor, mode = 'nearest',order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def worldToVoxelCoord(worldCoord, origin, spacing):

    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any( transformM!=np.array([1,0,0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing, isflip

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def savenpy_luna_attribute(inputpath, savepath):
    islabel = True
    isClean = True
    resolution = np.array([1, 1, 1])
    sliceim, origin, spacing, isflip = load_itk_image(os.path.join(luna_data, name + '.mhd'))
    ori_sliceim_shape_yx = sliceim.shape[1:3]
    if isflip:
        sliceim = sliceim[:, ::-1, ::-1]
        print('flip!')
    sliceim = lumTrans(sliceim)
    sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
    sliceim = sliceim1[np.newaxis, ...]
    np.save(os.path.join(savepath, name + '_clean.npy'), sliceim)
    return 1

def main():
    img_dir = "/research/dept8/jzwang/code/NoduleNet/dlung_v1/data"
    data_txt = "filedir.txt"
    lung_mask_dir = "./lung_mask"

    if not os.path.exists(lung_mask_dir):
        os.makedirs(lung_mask_dir)
    with open(os.path.join(img_dir, data_txt), "r") as f:
        lines = f.readlines()
    params_lists = []
    for line in lines:
        print("lung segmentation:", line)
        line = line.rstrip()
        savedir = '.'.join(line.split("/"))
        get_lung(os.path.join(img_dir, line), os.path.join(lung_mask_dir, savedir))

    """
    params_lists = []

    pool = Pool(processes=10)

    pool.map(savenpy_luna_attribute, params_lists)
    pool.close()
    pool.join()
    """
if __name__=='__main__':
    main()
