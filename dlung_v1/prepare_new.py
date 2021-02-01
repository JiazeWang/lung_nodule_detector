import os
import shutil
import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from scipy.ndimage.measurements import label
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from skimage import measure, morphology
from lungmask import mask
import pandas
import sys
import math
import glob
import nrrd
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
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def load_itk_dicom(filename):
    """Return img array and [z,y,x]-ordered origin and spacing
    """
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(filename)
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    numpyImage = sitk.GetArrayFromImage(img)
    numpyOrigin = np.array(list(reversed(img.GetOrigin())))
    numpySpacing = np.array(list(reversed(img.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing

def lumTrans(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype('uint8')

    return image_new


def convex_hull_dilate(binary_mask, dilate_factor=1.5, iterations=10):
    """
    Replace each slice with convex hull of it then dilate. Convex hulls used
    only if it does not increase area by dilate_factor. This applies mainly to
    the inferior slices because inferior surface of lungs is concave.
    binary_mask: 3D binary numpy array with the same shape of the image,
        that only region of interest is True. One side of the lung in this
        specifical case.
    dilate_factor: float, factor of increased area after dilation
    iterations: int, number of iterations for dilation
    return: 3D binary numpy array with the same shape of the image,
        that only region of interest is True. Each binary mask is ROI of one
        side of the lung.
    """
    binary_mask_dilated = np.array(binary_mask)
    for i in range(binary_mask.shape[0]):
        slice_binary = binary_mask[i]

        if np.sum(slice_binary) > 0:
            slice_convex = morphology.convex_hull_image(slice_binary)

            if np.sum(slice_convex) <= dilate_factor * np.sum(slice_binary):
                binary_mask_dilated[i] = slice_convex

    struct = scipy.ndimage.morphology.generate_binary_structure(3, 1)
    binary_mask_dilated = scipy.ndimage.morphology.binary_dilation(
        binary_mask_dilated, structure=struct, iterations=10)

    return binary_mask_dilated


def apply_mask(image, binary_mask1, binary_mask2, pad_value=170,
               bone_thred=210, remove_bone=False):
    """
    Apply the binary mask of each lung to the image. Regions out of interest
    are replaced with pad_value.
    image: 3D uint8 numpy array with the same shape of the image.
    binary_mask1: 3D binary numpy array with the same shape of the image,
        that only one side of lung is True.
    binary_mask2: 3D binary numpy array with the same shape of the image,
        that only the other side of lung is True.
    pad_value: int, uint8 value for padding image regions that is not
        interested.
    bone_thred: int, uint8 threahold value for determine parts of image is
        bone.
    return: D uint8 numpy array with the same shape of the image after
        applying the lung mask.
    """
    binary_mask = binary_mask1 + binary_mask2
    binary_mask1_dilated = convex_hull_dilate(binary_mask1)
    binary_mask2_dilated = convex_hull_dilate(binary_mask2)
    binary_mask_dilated = binary_mask1_dilated + binary_mask2_dilated
    binary_mask_extra = binary_mask_dilated ^ binary_mask

    # replace image values outside binary_mask_dilated as pad value
    image_new = image * binary_mask_dilated + \
        pad_value * (1 - binary_mask_dilated).astype('uint8')

    # set bones in extra mask to 170 (ie convert HU > 482 to HU 0;
    # water).
    if remove_bone:
        image_new[image_new * binary_mask_extra > bone_thred] = pad_value

    return image_new

def savenpy_luna_attribute(params_lists):
    inputpath, savepath, maskpath = params_lists
    islabel = True
    isClean = True
    resolution = np.array([1, 1, 1])
    sliceim, origin, spacing = load_itk_dicom(inputpath)
    lung_mask, _, _ = load_itk_image(maskpath)
    ori_sliceim_shape_yx = sliceim.shape[1:3]
    sliceim = lumTrans(sliceim)
    binary_mask1, binary_mask2 = lung_mask == 1, lung_mask == 2
    binary_mask = binary_mask1 + binary_mask2
    sliceim = apply_mask(sliceim, binary_mask1, binary_mask2)
    sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
    sliceim = sliceim1[np.newaxis, ...]
    np.save(os.path.join(savepath, name + '_clean.npy'), sliceim)
    nrrd.write(os.path.join(savepath, '%s_clean.nrrd' % (savename)), seg_img)
    return 1

def main():
    img_dir = "/research/dept8/jzwang/code/NoduleNet/dlung_v1/data"
    data_txt = "filedir.txt"
    lung_mask_dir = "./lung_mask"
    npy_dir = "./npy"
    if not os.path.exists(lung_mask_dir):
        os.makedirs(lung_mask_dir)
    with open(os.path.join(img_dir, data_txt), "r") as f:
        lines = f.readlines()
    params_lists = []
    for line in lines:
        print("lung segmentation:", line)
        line = line.rstrip()
        savedir = '_'.join(line.split("/"))
        get_lung(os.path.join(img_dir, line), os.path.join(lung_mask_dir, savedir))
    params_lists = []
    for line in lines:
        line = line.rstrip()
        savename = '_'.join(line.split("/"))
        npy_savepath = os.path.join(npy_dir, savename)
        mask_savepath =  os.path.join(lung_mask_dir, savename+'.mhd')
        params_lists.append([os.path.join(img_dir, line), npy_savepath, mask_savepath])
    for line in params_lists:
        savenpy_luna_attribute(line)
    pool = Pool(processes=10)
    pool.map(savenpy_luna_attribute, params_lists)
    pool.close()
    pool.join()

if __name__=='__main__':
    main()
