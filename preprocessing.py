#computes necessary pre-processing for CT imaging
# 1) creates CT body mask
# 2) resamples images to 1x1x5
# 3) creates windowed/normalized imaging

#FIX ITK PARALLELIZATION!

import os
import itk
import numpy as np
import nibabel as nib
import multiprocessing

from skimage import filters
from joblib import Parallel, delayed
from _preprocessing_functions import *
from skimage.measure import regionprops
from scipy.ndimage import label as scipy_conn_comp
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure, binary_fill_holes, binary_closing

#
base_dir = '/root/jay.patel@ccds.io/mnt/2015P002510/Jay/Projects/Adrenal/'
folders = ['Train/', 'Val/', 'Test/']
volsToProcess = ['ct_vol.nii.gz']
roisToProcess = ['adrenal-label.nii.gz', 'adrenal_bcb23-label.nii.gz', 'adrenal_wm37-label.nii.gz', 'adrenal_dig1-label.nii.gz', 'adrenal_bd47-label.nii.gz', 'adrenal_cr77-label.nii.gz']

#path to SLICER3D (used for reorienting, registering, resampling volumes)
slicer_dir = '/home/shared_software/Slicer-4.10.2-linux-amd64/Slicer'

mask_name = 'ct_body_mask-label.nii.gz'
outputVolNames = ['ct_all.nii.gz', 'ct_soft_tissue.nii.gz', 'ct_liver.nii.gz', 'ct_subdural.nii.gz']
outputROINames = ['adrenal_cr77_binary-label.nii.gz', 'adrenal_bcb23_binary-label.nii.gz', 'adrenal_wm37_binary-label.nii.gz', 'adrenal_dig1_binary-label.nii.gz', 'adrenal_bd47_binary-label.nii.gz', 'adrenal_cr77_binary-label.nii.gz']
interp_type_vol = 'bspline'
interp_type_roi = 'nearestNeighbor'
lower_threshold_num = -500
num_iter_erode_dilate = 3
global_window_mean_std = np.array([[-33.8201, 244.97824], [1.77538315, 85.65067226], [56.046277, 58.28176293], [47.80042306, 33.21754168]])
level_width = np.array([[np.nan, np.nan], [70, 370], [96, 276], [50, 130]])
intensity_range = np.array([[level - width/2, level + width/2] for level,width in level_width])
num_channels = intensity_range.shape[0]
resampled_suffix = '_RESAMPLED.nii.gz'
new_spacing = [1,1,5]
spacing_slicer = '1,1,5'

def get_largest_component(mask, se, vol):
    conn_comp, cc_num = scipy_conn_comp(mask, se)
    conn_comp_properties = regionprops(conn_comp, vol)
    largest_componenet_size = 0
    largest_componenet_index = 0
    for i in range(0, len(conn_comp_properties)):
        temp_area = conn_comp_properties[i].area
        if temp_area > largest_componenet_size:
            largest_componenet_size = temp_area
            largest_componenet_index = i
    #
    mask = conn_comp == (largest_componenet_index+1)
    return mask

def generate_ct_mask(patient, folder):
    os.chdir(base_dir + folder + patient)
    #load in volume
    vol_nib = nib.load(volsToProcess[0])
    vol = vol_nib.get_data()
    affine = vol_nib.get_affine()
    header = vol_nib.get_header()
    #otsu threshold at lower bound
    #val = filters.threshold_otsu(vol[vol>lower_threshold_num])
    #mask = vol > val
    mask = vol > lower_threshold_num
    #get largest connected component
    se = generate_binary_structure(len(vol.shape), len(vol.shape))
    mask = get_largest_component(mask, se, vol)
    #erode mask a couple of times to get rid of table
    mask = binary_erosion(mask, se, num_iter_erode_dilate)
    #dilate mask to re-gain border
    mask = binary_dilation(mask, se, num_iter_erode_dilate)
    #fill holes in 3 different axes to account for broken surface in 3D
    mask_pre_filter = np.copy(mask)
    converged = False
    while converged == False:
        for i in range(0, mask.shape[0]):
            mask[i,:,:] = binary_fill_holes(mask[i,:,:])
        for i in range(0, mask.shape[1]):
            mask[:,i,:] = binary_fill_holes(mask[:,i,:])
        for i in range(0, mask.shape[2]):
            mask[:,:,i] = binary_fill_holes(mask[:,:,i])
        if np.all(mask_pre_filter == mask):
            converged = True
        else:
            mask_pre_filter = np.copy(mask)
    #apply closing to fix small imperfections of surface (closing affects top and bottom slices so ensure to replace those)
    mask_top = np.copy(mask[:,:,0])
    mask_bottom = np.copy(mask[:,:,-1])
    mask = binary_closing(mask)
    mask[:,:,0] = mask_top
    mask[:,:,-1] = mask_bottom
    #get largest connected component in case something accidently got added
    mask = get_largest_component(mask, se, vol)
    #save mask output
    nib_mask = nib.Nifti1Image(mask.astype(int), affine, header=header)
    nib.save(nib_mask, mask_name)

for folder in folders:
    patient_dir = base_dir + folder
    patients = nested_folder_filepaths(patient_dir, volsToProcess)
    patients.sort()
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(generate_ct_mask)(patient, folder) for patient in patients)

#resample all images
def resample_itk(patient, folder, input_image_name, new_spacing=[1,1,1], interpolator_function='BSpline', scale=[1,1,1]):
    #interpolator_function = 'BSpline'   #choose from BSpline, Linear, NearestNeighbor
    os.chdir(base_dir + folder + patient)
    output_image_name = input_image_name[:-7] + resampled_suffix
    vol = nib.load(input_image_name).get_data()
    #set types
    PixelType = itk.F
    ScalarType1 = itk.D
    ScalarType2 = itk.F
    Dimension = len(vol.shape)
    ImageType = itk.Image[PixelType, Dimension]
    #read in image to itk
    ReaderType = itk.ImageFileReader[ImageType]
    reader = ReaderType.New()
    reader.SetFileName(input_image_name)
    reader.Update()
    input_image = reader.GetOutput()
    #set output variables
    size = input_image.GetLargestPossibleRegion().GetSize()
    spacing = input_image.GetSpacing()
    origin = input_image.GetOrigin()
    direction = input_image.GetDirection()
    new_size = np.array(size) * spacing / new_spacing
    new_size = list(np.ceil(new_size).astype(np.int)) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    #set center of volume
    centralPixel = itk.Index[Dimension]()
    for i in range(0, len(size)):
        centralPixel[i] = int(size[i] / 2)
    centralPoint = itk.Point[ScalarType1, Dimension]()
    for i in range(0, len(centralPixel)):
        centralPoint[i] = centralPixel[i]
    #initialize scale transform
    scaleTransform = itk.ScaleTransform[ScalarType1, Dimension].New()
    parameters = scaleTransform.GetParameters()
    for i in range(0, Dimension):
        parameters[i] = scale[i]
    scaleTransform.SetParameters(parameters)
    scaleTransform.SetCenter(centralPoint)
    #set interpolator
    if interpolator_function == 'BSpline':
        interpolatorType = itk.BSplineInterpolateImageFunction[ImageType, ScalarType1, ScalarType2]
    elif interpolator_function == 'Linear':
        interpolatorType = itk.LinearInterpolateImageFunction[ImageType, ScalarType1]
    else:
        interpolatorType = itk.NearestNeighborInterpolateImageFunction[ImageType, ScalarType1]
    #
    interpolator = interpolatorType.New()
    #initialize resample filter
    resamplerType = itk.ResampleImageFilter[ImageType, ImageType]
    resampleFilter = resamplerType.New()
    #set variables
    resampleFilter.SetInput(input_image)
    resampleFilter.SetTransform(scaleTransform)
    resampleFilter.SetInterpolator(interpolator)
    resampleFilter.SetSize(new_size)
    resampleFilter.SetOutputSpacing(new_spacing)
    resampleFilter.SetOutputOrigin(origin)
    resampleFilter.SetOutputDirection(direction)
    #execute filter
    WriterType = itk.ImageFileWriter[ImageType]
    writer = WriterType.New()
    writer.SetFileName(output_image_name)
    writer.SetInput(resampleFilter.GetOutput())
    writer.Update()

#resample images
def resample_slicer(patient):
    resampled_volumes = resample_volume(patient_dir, patient, volsToProcess, spacing_slicer, interp_type_vol, slicer_dir)
    resample_volume_using_reference(patient_dir, patient, [mask_name], resampled_volumes[0], interp_type_roi, slicer_dir)
    if len(roisToProcess) > 0:
        for roi_to_process in roisToProcess:
            if os.path.exists(patient_dir + patient + '/' + roi_to_process):
                resample_volume_using_reference(patient_dir, patient, [roi_to_process], resampled_volumes[0], interp_type_roi, slicer_dir)

for folder in folders:
    patient_dir = base_dir + folder
    patients = nested_folder_filepaths(patient_dir, volsToProcess)
    patients.sort()
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(resample_slicer)(patient) for patient in patients)
    #Parallel(n_jobs=num_cores, backend='threading')(delayed(resample_itk)(patient, folder, volsToProcess[0], new_spacing=new_spacing, interpolator_function='BSpline', scale=[1,1,1]) for patient in patients)
    #if len(roisToProcess) > 0:
    #    Parallel(n_jobs=num_cores, backend='threading')(delayed(resample_itk)(patient, folder, roisToProcess[0], new_spacing=new_spacing, interpolator_function='NearestNeighbor', scale=[1,1,1]) for patient in patients)
    #Parallel(n_jobs=num_cores, backend='threading')(delayed(resample_itk)(patient, folder, mask_name, new_spacing=new_spacing, interpolator_function='NearestNeighbor', scale=[1,1,1]) for patient in patients)
    #for patient in patients:
    #    resample_itk(patient, folder, volsToProcess[0], new_spacing=new_spacing, interpolator_function='BSpline', scale=[1,1,1])
    #    resample_itk(patient, folder, mask_name, new_spacing=new_spacing, interpolator_function='NearestNeighbor', scale=[1,1,1])

#Create windowed images and normalize images using global mean and standard deviation
#binarize mask and one hot encode
def normalize(patient, folder):
    os.chdir(base_dir + folder + patient)
    #if not np.all([os.path.exists(outputVolName) for outputVolName in outputVolNames[:-1]]):
    #load in volume
    tempVol_nib = nib.load(volsToProcess[0][:-7] + resampled_suffix)
    affine = tempVol_nib.affine
    header = tempVol_nib.header
    #Normalize image (make zero mean and standard deviation of 1)
    CT_vol = tempVol_nib.get_data().astype(np.float32)
    ct_mask = nib.load(mask_name[:-7] + resampled_suffix).get_data()
    for i in range(num_channels):
        lower, upper = intensity_range[i,:]
        if np.all(np.isnan([lower, upper])):
            idx_window = np.where(ct_mask == 1)
        else:
            idx_window = np.where((CT_vol >= lower) & (CT_vol <= upper) & (ct_mask == 1))
        temp_ct_window = np.copy(CT_vol)
        #normalize inside the window and set outside the window equal to either min or max value
        temp_ct_window[idx_window] = (temp_ct_window[idx_window] - global_window_mean_std[i,0]) / global_window_mean_std[i,1]
        if not np.all(np.isnan([lower, upper])):
            temp_ct_window[np.where(CT_vol < lower)] = np.min(temp_ct_window[idx_window])
            temp_ct_window[np.where(CT_vol > upper)] = np.max(temp_ct_window[idx_window])
        #mask out volume
        temp_ct_window = temp_ct_window * ct_mask
        #save normalized volume
        CT_window_nib = nib.Nifti1Image(temp_ct_window, affine, header=header)
        nib.save(CT_window_nib, base_dir + folder + patient + '/' + outputVolNames[i])
    #binarize label map
    if len(roisToProcess) > 0:
        for i, roi_to_process in enumerate(roisToProcess):
            if os.path.exists(base_dir + folder + patient + '/' + roi_to_process):
                tempLabel_nib = nib.load(roi_to_process[:-7] + resampled_suffix)
                affine = tempLabel_nib.affine
                header = tempLabel_nib.header
                tempLabel_binary = tempLabel_nib.get_data()
                tempLabel_binary[tempLabel_binary > 0] = 1
                #save binarized label
                tempLabel_binary_nib = nib.Nifti1Image(tempLabel_binary, affine, header=header)
                nib.save(tempLabel_binary_nib, base_dir + folder + patient + '/' + outputROINames[i])

for folder in folders:
    patient_dir = base_dir + folder
    patients = nested_folder_filepaths(patient_dir, volsToProcess)
    patients.sort()
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(normalize)(patient, folder) for patient in patients)