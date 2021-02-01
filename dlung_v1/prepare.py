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
from multiprocessing import Pool
from config import config

def get_lung(filename, output):
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(filename)
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    segmentation = mask.apply(img)
    result_out= sitk.GetImageFromArray(segmentation)
    output = output+'.mhd'
    sitk.WriteImage(result_out, output)

def resample(image, spacing, new_spacing=[1.0, 1.0, 1.0], order=1):
    """
    Resample image from the original spacing to new_spacing, e.g. 1x1x1
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    spacing: float * 3, raw CT spacing in [z, y, x] order.
    new_spacing: float * 3, new spacing used for resample, typically 1x1x1,
        which means standardizing the raw CT with different spacing all into
        1x1x1 mm.
    order: int, order for resample function scipy.ndimage.interpolation.zoom
    return: 3D binary numpy array with the same shape of the image after,
        resampling. The actual resampling spacing is also returned.
    """
    # shape can only be int, so has to be rounded.
    new_shape = np.round(image.shape * spacing / new_spacing)

    # the actual spacing to resample.
    resample_spacing = spacing * image.shape / new_shape

    resize_factor = new_shape / image.shape

    image_new = scipy.ndimage.interpolation.zoom(image, resize_factor,
                                                 mode='nearest', order=order)

    return (image_new, resample_spacing)

def get_lung_box(binary_mask, new_shape, margin=5):
    """
    Get the lung barely surrounding the lung based on the binary_mask and the
    new_spacing.
    binary_mask: 3D binary numpy array with the same shape of the image,
        that only region of both sides of the lung is True.
    new_shape: tuple of int * 3, new shape of the image after resamping in
        [z, y, x] order.
    margin: int, number of voxels to extend the boundry of the lung box.
    return: 3x2 2D int numpy array denoting the
        [z_min:z_max, y_min:y_max, x_min:x_max] of the lung box with respect to
        the image after resampling.
    """
    # list of z, y x indexes that are true in binary_mask
    z_true, y_true, x_true = np.where(binary_mask)
    old_shape = binary_mask.shape

    lung_box = np.array([[np.min(z_true), np.max(z_true)],
                        [np.min(y_true), np.max(y_true)],
                        [np.min(x_true), np.max(x_true)]])
    lung_box = lung_box * 1.0 * \
        np.expand_dims(new_shape, 1) / np.expand_dims(old_shape, 1)
    lung_box = np.floor(lung_box).astype('int')

    z_min, z_max = lung_box[0]
    y_min, y_max = lung_box[1]
    x_min, x_max = lung_box[2]

    # extend the lung_box by a margin
    lung_box[0] = max(0, z_min-margin), min(new_shape[0], z_max+margin)
    lung_box[1] = max(0, y_min-margin), min(new_shape[1], y_max+margin)
    lung_box[2] = max(0, x_min-margin), min(new_shape[2], x_max+margin)

    return lung_box

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
    reader = sitk.ImageSeriesReader()
    dcm_series = reader.GetGDCMSeriesFileNames(filename)
    reader.SetFileNames(dcm_series)
    img = reader.Execute()
    numpyImage = sitk.GetArrayFromImage(img)
    numpyOrigin = np.array(list(reversed(img.GetOrigin())))
    numpySpacing = np.array(list(reversed(img.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def load_itk_series(filename):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(filename)
    print("seriesIDs:", seriesIDs)
    print("len seriesIDs:", len(seriesIDs))
    dcm_series = reader.GetGDCMSeriesFileNames(filename, seriesIDs[0])
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
    print("Save %s to numpy"%inputpath)
    islabel = True
    isClean = True
    resolution = np.array([1, 1, 1])
    sliceim, origin, spacing = load_itk_dicom(inputpath)
    lung_mask, _, _ = load_itk_image(maskpath)
    np.save(savepath + '_origin.npy', origin)
    np.save(savepath + '_spacing.npy', spacing)
    binary_mask1, binary_mask2 = lung_mask == 1, lung_mask == 2
    binary_mask = binary_mask1 + binary_mask2
    ori_sliceim_shape_yx = sliceim.shape[1:3]
    sliceim = lumTrans(sliceim)
    sliceim = apply_mask(sliceim, binary_mask1, binary_mask2)
    sliceim1, _ = resample(sliceim, spacing, resolution, order=1)
    seg_img = sliceim1
    lung_box = get_lung_box(binary_mask, seg_img.shape)
    z_min, z_max = lung_box[0]
    y_min, y_max = lung_box[1]
    x_min, x_max = lung_box[2]
    seg_img = seg_img[z_min:z_max, y_min:y_max, x_min:x_max]
    #sliceim = sliceim1[np.newaxis, ...]
    np.save(savepath + '_clean.npy', seg_img)
    #nrrd.write(savepath + '_clean.nrrd', seg_img)
    return 1

def main():
    img_dir = config["img_dir"]
    data_txt = config["data_txt"]
    lung_mask_dir = config["lung_mask_dir"]
    npy_dir = config["npy_dir"]
    if not os.path.exists(lung_mask_dir):
        os.makedirs(lung_mask_dir)
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)
    with open(data_txt, "r") as f:
        lines = f.readlines()
    params_lists = []
    for line in lines:
        #print("lung segmentation:", line)
        line = line.rstrip()
        savedir = '_'.join(line.split("/"))
        numpyImage, numpyOrigin, numpySpacing = load_itk_dicom(os.path.join(img_dir, line))
    """
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
    pool = Pool(processes=10)
    pool.map(savenpy_luna_attribute, params_lists)
    pool.close()
    pool.join()
    """

if __name__=='__main__':
    main()
