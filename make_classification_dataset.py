import os
import numpy as np
import pandas as pd
import nibabel as nib

from _preprocessing_functions import *
from scipy.ndimage import find_objects
from skimage.measure import regionprops
from scipy.ndimage import label as scipy_conn_comp
from scipy.ndimage import binary_dilation, generate_binary_structure

#
base_dir = '/root/jay.patel@ccds.io/mnt/2015P002510/Jay/Projects/Adrenal/'
folders = ['Train/', 'Val/', 'Test/']

input_vols = ['ct_all.nii.gz', 'ct_liver.nii.gz', 'ct_soft_tissue.nii.gz', 'ct_subdural.nii.gz']
input_pred_seg = ['model_final_processed-label.nii.gz']

resize_adrenal_shape = (80,80,24)
se = generate_binary_structure(3,3)
num_iter_dilate_postprocess = 3      #might be too many (possibly change this down to 1 or 2)
voxels_gap_between_adrenals = 3      #minimum gap between adrenals for when making crops

def make_feature_matrix(patients):
    crop_im_list_ref_pred = []
    crop_seg_pred_list_ref_pred = []
    for patient in patients:
        print(patient)
        os.chdir(nifti_dir + patient)
        #load in all images and segs
        im = np.stack([nib.load(input_vol).get_data() for input_vol in input_vols])
        patient_pred_seg = [input_seg_patient for input_seg_patient in input_pred_seg if os.path.exists(input_seg_patient)]
        seg_pred = (nib.load(patient_pred_seg[0]).get_data() > 0).astype(int)
        #
        cropped_left_im_ref_pred, cropped_right_im_ref_pred, cropped_left_seg_pred_ref_pred, cropped_right_seg_pred_ref_pred = process_patient(im, seg_pred)
        #
        crop_im_list_ref_pred.append(cropped_left_im_ref_pred)
        crop_im_list_ref_pred.append(cropped_right_im_ref_pred)
        crop_seg_pred_list_ref_pred.append(cropped_left_seg_pred_ref_pred)
        crop_seg_pred_list_ref_pred.append(cropped_right_seg_pred_ref_pred)
    #
    images_matrix_ref_pred = np.stack(crop_im_list_ref_pred)
    segs_pred_matrix_ref_pred = np.stack(crop_seg_pred_list_ref_pred)
    return images_matrix_ref_pred, segs_pred_matrix_ref_pred

def process_patient(im, seg_pred):
    reference_seg = seg_pred
    if np.any(reference_seg):
        volume_location = find_objects(reference_seg != 0)[0]
    else:
        volume_location = tuple([slice(0, dim_shape) for dim_shape in reference_seg.shape])
    # only grab the largest component of the CT scan in the vicinity of the adrenal (to correct for edge cases like Test/Dataset_4/2.25.264988446360727031086343065799006710528)
    im_binary = np.copy(im)
    im_binary[...,:volume_location[2].start] = 0
    im_binary[...,volume_location[2].stop:] = 0
    im_binary = np.squeeze((im_binary != 0).astype(np.int)[0,...])
    conn_comp, cc_num = scipy_conn_comp(im_binary, se)
    conn_comp_properties = regionprops(conn_comp, im_binary)
    conn_comp_areas = [i.area for i in conn_comp_properties]
    sorted_components = np.argsort(conn_comp_areas)
    im_binary_largest_comp = (conn_comp == (sorted_components[-1]+1)).astype(np.int)
    im_filtered = im * np.expand_dims(im_binary_largest_comp,axis=0)
    volume_location = find_objects(im_filtered != 0)[0]
    seg_half_index = ((volume_location[1].stop - volume_location[1].start) // 2) + volume_location[1].start
    # split mask into left/right adrenal
    reference_seg_dilated = binary_dilation(reference_seg, se, num_iter_dilate_postprocess).astype(int)
    conn_comp, cc_num = scipy_conn_comp(reference_seg_dilated, se)
    if cc_num > 0:
        conn_comp_properties = regionprops(conn_comp, reference_seg_dilated)
        conn_comp_areas = [i.area for i in conn_comp_properties]
        sorted_components = np.argsort(conn_comp_areas)
        if cc_num == 1:
            reference_seg_dilated = (1*(conn_comp == (sorted_components[-1]+1)).astype(np.int))
        else:
            reference_seg_dilated = (1*(conn_comp == (sorted_components[-1]+1)).astype(np.int)) + (2*(conn_comp == (sorted_components[-2]+1)).astype(np.int))
        final_split_mask = reference_seg_dilated * reference_seg
    else:
        final_split_mask = reference_seg
    # if mask is completely empty, use centerline to make crops
    if np.any(final_split_mask) == False:
        split_left = seg_half_index
        split_right = seg_half_index
    # if mask only has one adrenal, need to determine where to crop image based on one adrenal
    elif len(np.unique(final_split_mask)) == 2:
        volume_location = find_objects(final_split_mask != 0)[0]
        #this is left adrenal since more of the adrenal is on left side
        if np.sum(final_split_mask[seg_half_index:,...]) > np.sum(final_split_mask[:seg_half_index,...]):
            split_left = min(seg_half_index, volume_location[0].start - resize_adrenal_shape[0])
            split_right = max(seg_half_index, volume_location[0].start - voxels_gap_between_adrenals)
        #this is right adrenal since more of the adrenal is on right side
        else:
            split_left = min(seg_half_index, volume_location[0].stop + voxels_gap_between_adrenals)
            split_right = max(seg_half_index, volume_location[0].stop + resize_adrenal_shape[0])
    # if mask has two adrenals, can optimize for both to ensure as little zero-padding as necessary
    else:
        for i in range(0, final_split_mask.shape[0]):
            if np.any(final_split_mask[i,...]):
                adrenal_label = np.unique(final_split_mask[i,...])[-1]
                break
        volume_location_left = find_objects(final_split_mask == adrenal_label)[0]
        #
        for i in range(final_split_mask.shape[0]-1, -1, -1):
            if np.any(final_split_mask[i,...]):
                adrenal_label = np.unique(final_split_mask[i,...])[-1]
                break
        volume_location_right = find_objects(final_split_mask == adrenal_label)[0]
        split_left = min(volume_location_right[0].start - voxels_gap_between_adrenals, volume_location_left[0].stop + voxels_gap_between_adrenals)
        split_right = max(volume_location_left[0].stop + voxels_gap_between_adrenals, volume_location_right[0].start - voxels_gap_between_adrenals)
    #
    final_reference_seg = (final_split_mask > 0).astype(np.int)
    left_im = np.copy(im)
    left_im = left_im[:,split_left:, ...]
    right_im = np.copy(im)
    right_im = right_im[:,:split_right, ...]
    #
    left_seg_pred = np.copy(final_reference_seg)
    left_seg_pred = left_seg_pred[split_left:, ...]
    right_seg_pred = np.copy(final_reference_seg)
    right_seg_pred = right_seg_pred[:split_right, ...]
    #crop images/seg
    cropped_left_im, cropped_left_seg_pred = crop_seg(left_im, left_seg_pred, resize_adrenal_shape)
    cropped_right_im, cropped_right_seg_pred = crop_seg(right_im, right_seg_pred, resize_adrenal_shape)
    #
    return cropped_left_im, cropped_right_im, cropped_left_seg_pred, cropped_right_seg_pred

def crop_seg(im, seg_pred, resize_adrenal_shape):
    im = np.pad(im, [tuple((0,0))] +[(x,x) for x in resize_adrenal_shape], 'constant')
    seg_pred_pad = np.pad(seg_pred, [(x,x) for x in resize_adrenal_shape], 'constant')
    seg_reference = seg_pred_pad
    if np.any(seg_reference):
        volume_location = find_objects(seg_reference)[0]
        #crop large enough size so that no resizing needed
        volume_location_crop = tuple([slice(int(volume_location[i].start - np.floor((resize_adrenal_shape[i] - (volume_location[i].stop - volume_location[i].start)) / 2)), int(volume_location[i].stop + np.ceil((resize_adrenal_shape[i] - (volume_location[i].stop - volume_location[i].start)) / 2))) for i in range(0, len(volume_location))])
        cropped_seg_pred = np.copy(seg_pred_pad[volume_location_crop])
        cropped_im = np.copy(im[tuple(list((slice(None,None),)) + list(volume_location_crop))])
    else:
        cropped_seg_pred = np.zeros(resize_adrenal_shape)
        cropped_im = np.zeros((im.shape[0],) + resize_adrenal_shape)
    return cropped_im, cropped_seg_pred

def filter_cases(images_list_ref_pred, segs_pred_list_ref_pred, patients_list):
    final_images_list_ref_pred = []
    final_segs_pred_list_ref_pred = []
    final_patients_list = []
    final_left_right_adrenal_list = []
    for temp_images_ref_pred, temp_segs_pred_ref_pred, temp_patients in zip(images_list_ref_pred, segs_pred_list_ref_pred, patients_list):
        temp_images_list_ref_pred = []
        temp_segs_pred_list_ref_pred = []
        temp_patients_list = []
        temp_left_right_list = []
        for i in range(0, temp_images_ref_pred.shape[0]):
            #check if the label should be kept, and that the image is not completely empty
            if np.any(temp_images_ref_pred[i,...]):
                temp_images_list_ref_pred.append(temp_images_ref_pred[i,...].astype(np.float32))
                temp_segs_pred_list_ref_pred.append(temp_segs_pred_ref_pred[i,...].astype(np.float32))
                temp_patients_list.append(temp_patients[i//2])
                if i % 2 == 0:
                    temp_left_right_list.append('left')
                else:
                    temp_left_right_list.append('right')
        final_images_list_ref_pred.append(np.stack(temp_images_list_ref_pred))
        final_segs_pred_list_ref_pred.append(np.stack(temp_segs_pred_list_ref_pred))
        final_patients_list.append(temp_patients_list)
        final_left_right_adrenal_list.append(temp_left_right_list)
    return final_images_list_ref_pred, final_segs_pred_list_ref_pred, final_patients_list, final_left_right_adrenal_list

images_list_ref_pred = []
segs_pred_list_ref_pred = []
patients_list = []
for folder in folders:
    nifti_dir = base_dir + folder
    patients = nested_folder_filepaths(nifti_dir, input_vols)
    patients.sort()
    images_matrix_ref_pred, segs_pred_matrix_ref_pred = make_feature_matrix(patients)
    #
    images_list_ref_pred.append(np.transpose(np.stack(images_matrix_ref_pred), (0,2,3,4,1)))
    segs_pred_list_ref_pred.append(np.expand_dims(segs_pred_matrix_ref_pred, axis=-1))
    #
    patients_list.append([folder + patient for patient in patients])

#remove cases that were not segmented (will be counted as absent for purpose of accounting)
final_images_list_ref_pred, final_segs_pred_list_ref_pred, final_patients_list, final_left_right_adrenal_list = filter_cases(images_list_ref_pred, segs_pred_list_ref_pred, patients_list)

#save to individual nifti_files
save_seg_names = ['predicted_ref_pred_crop-label.nii.gz']
for i, patient_list in enumerate(final_patients_list):
    for j, patient in enumerate(patient_list):
        patient_dir = base_dir + patient + '/'
        crop_dir = patient_dir + final_left_right_adrenal_list[i][j] + '/'
        if not os.path.exists(crop_dir):
            os.makedirs(crop_dir)
        #save all files as nifti to crop dir
        ref_image_nib = nib.load(patient_dir + input_vols[0])
        affine = ref_image_nib.get_affine()
        header = ref_image_nib.get_header()
        for k in range(0, len(input_vols)):
            temp_crop_name = input_vols[k][:-7] + '_ref_pred_crop.nii.gz'
            nib.save(nib.Nifti1Image(final_images_list_ref_pred[i][j,...,k], affine=affine, header=header), crop_dir + temp_crop_name)
        nib.save(nib.Nifti1Image(final_segs_pred_list_ref_pred[i][j,...,0], affine=affine, header=header), crop_dir + save_seg_names[0])
