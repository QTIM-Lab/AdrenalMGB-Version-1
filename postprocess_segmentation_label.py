import os
import numpy as np
import nibabel as nib
import multiprocessing

from joblib import Parallel, delayed
from _preprocessing_functions import *
from skimage.measure import regionprops
from scipy.ndimage import label as scipy_conn_comp
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure

base_dir = '/root/jay.patel@ccds.io/mnt/2015P002510/Jay/Projects/Adrenal/'
folders = ['Train/', 'Val/', 'Test/']
model_label = 'model-label.nii.gz'
save_name = ['model_final-label.nii.gz', 'model_final_processed-label.nii.gz']
input_ct_vol = 'ct_soft_tissue.nii.gz'
reference_ct = 'ct_vol.nii.gz'
se = generate_binary_structure(3,3)
num_iter_dilate_postprocess = 3
interp_type_roi = 'nearestNeighbor'

#path to SLICER3D (used for reorienting, registering, resampling volumes)
slicer_dir = '/home/shared_software/Slicer-4.10.2-linux-amd64/Slicer'

def ensemble_labels(patient):
    os.chdir(patient)
    #load in volumes
    vol_nib = nib.load(model_label)
    affine = vol_nib.get_affine()
    header = vol_nib.get_header()
    ensembled_label = vol_nib.get_data()
    im = nib.load(input_ct_vol).get_data()
    #postprocess label to remove extraneous components
    ensembled_label_postprocessed = postprocess_ensembled_label(ensembled_label, im)
    #save final labels out
    nib_ensemble1 = nib.Nifti1Image(ensembled_label, affine, header=header)
    nib.save(nib_ensemble1, patient + '/' + save_name[0])
    nib_ensemble2 = nib.Nifti1Image(ensembled_label_postprocessed, affine, header=header)
    nib.save(nib_ensemble2, patient + '/' + save_name[1])

def postprocess_ensembled_label(ensembled_label, im):
    #dilate label in case adrenal is split into multiple components
    ensembled_label_binary = (ensembled_label * (im != 0)).astype(int)
    ensembled_label_dilated = binary_dilation(ensembled_label_binary, se, num_iter_dilate_postprocess).astype(int)
    conn_comp, cc_num = scipy_conn_comp(ensembled_label_dilated, se)
    #erode image and get border to find if there is anything segmented near the border of the image
    im_binary = np.copy(im)
    im_binary = (im_binary != 0).astype(np.int)
    im_binary = binary_fill_holes(im_binary)
    im_binary_eroded = binary_erosion(im_binary, se, num_iter_dilate_postprocess, border_value=1).astype(int)
    border_region = (np.logical_xor(im_binary, im_binary_eroded)).astype(np.int)
    conn_comp_border = conn_comp * border_region
    unique_comps = np.unique(conn_comp_border)
    for unique_comp in unique_comps:
        conn_comp[conn_comp == unique_comp] = 0
    conn_comp_eroded = (conn_comp > 0).astype(int)
    ensembled_label_eroded = ensembled_label_dilated * conn_comp_eroded
    conn_comp, cc_num = scipy_conn_comp(ensembled_label_eroded, se)
    if cc_num > 0:
        conn_comp_properties = regionprops(conn_comp, ensembled_label_dilated)
        conn_comp_areas = [i.area for i in conn_comp_properties]
        sorted_components = np.argsort(conn_comp_areas)
        #if only one predicted components, take that one
        if cc_num == 1:
            pred_seg_dilated = (conn_comp == (sorted_components[-1]+1)).astype(np.int)
        #if 2 or more predicted components, rank order by size and take largest
        else:
            pred_seg_dilated = ((conn_comp == (sorted_components[-1]+1)) | ((conn_comp == (sorted_components[-2]+1)))).astype(np.int)
        final_mask = pred_seg_dilated * ensembled_label_binary
    else:
        final_mask = ensembled_label_binary
    return final_mask

for folder in folders:
    patient_dir = base_dir + folder
    patients = nested_folder_filepaths(patient_dir, [model_label])
    patients.sort()
    patients = [patient_dir + patient for patient in patients]
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(ensemble_labels)(patient) for patient in patients)

def resample_back_to_original_spacing(patient, final_label, reference_im):
    resample_volume_using_reference(patient_dir, patient, [final_label], reference_im, interp_type_roi, slicer_dir)

for folder in folders:
    patient_dir = base_dir + folder
    patients = nested_folder_filepaths(patient_dir, [save_name[1]])
    patients.sort()
    patients = [patient_dir + patient for patient in patients]
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(resample_back_to_original_spacing)(patient, save_name[1], reference_ct) for patient in patients)