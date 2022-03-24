#TODO: Add more test-time augmentation ability (multi-scale testing? gamma corrected inputs?)
#TODO: Cleaner way of saving probability masks and binarized outputs at end
#TODO: Load certain model specific parameters from the model config file
#TODO: Do not apply scipy.zoom on all maps since it is a very time-consuming process; instead, combine the maps first and then scipy.zoom the combined map 
#TODO: Save out uncertainty maps (both using averaged probability and entropy)

import time
import numpy as np
import nibabel as nib
import tensorflow as tf

from unet import load_my_model
from sorcery import unpack_keys
from scipy.ndimage import find_objects
from itertools import product, combinations
from scipy.ndimage import zoom as scipy_zoom
from load_data import nested_folder_filepaths
from scipy.ndimage.interpolation import map_coordinates

def predict_model(params_dict):
    #unpack relevant dictionary elements
    output_file, data_dirs_predict, input_image_names = unpack_keys(params_dict)
    #load model
    model, manager, ckpt = load_my_model(params_dict)
    #list patients to test
    patients = []
    for data_dir in data_dirs_predict:
        #patient list
        relative_filepaths = nested_folder_filepaths(data_dir, [input_image_names])
        dir_patients = [data_dir + relative_filepath + '/' for relative_filepath in relative_filepaths]
        patients.extend(dir_patients)
    num_patients = len(patients)
    for counter, patient_path in enumerate(patients):
        start_prediction_time = time.time()
        predict_volume(model, manager, ckpt, patient_path, params_dict)
        time_for_prediction = int(time.time() - start_prediction_time)
        with open(output_file, 'a') as f:
            f.write(str(counter+1) + '/' + str(num_patients) + ': Prediction for patient ' + patient_path + ' is complete (' + str(time_for_prediction) + 's) \n')

#Function to patch test image based on overlap amount and boundary removal
def predict_volume(model, manager, ckpt, patient_path, params_dict):
    #unpack relevant dictionary elements
    input_image_names, network_name, predict_using_patches, predict_fast_to_find_roi, predicted_label_name, patch_size, save_uncertainty_map, num_outputs, binarize_value = unpack_keys(params_dict)
    #load in patient volume
    all_input_vols_orig, affine, header = load_volumes_nibabel(patient_path, input_image_names)
    #crop out zero values to improve throughput
    foreground_image = np.sum(all_input_vols_orig, axis=-1)
    if np.any(foreground_image != 0):
        volume_location = find_objects(foreground_image != 0)[0]
    else:
        volume_location = find_objects(foreground_image == 0)[0]
    all_input_vols = all_input_vols_orig[volume_location]
    #pad all axes to the patch size in case cropped volume is smaller than size of patch along an axis or axes
    extra_padding = np.maximum(patch_size - np.array(all_input_vols.shape[:-1]), 0)
    pad_tuple_initial = tuple([(int(np.floor(i / 2)), int(np.ceil(i / 2))) for i in extra_padding] + [(0,0)])
    all_input_vols = np.pad(all_input_vols, pad_tuple_initial, mode='constant')
    #Different inference procedures for segmentation and classification
    if network_name == 'Unet':
        #predict output volume by either predicting patches and combining, or by passing the entire volume to be segmented at once
        if predict_using_patches == True:
            roi_location = None
            if predict_fast_to_find_roi[0] == True:
                #predict using small patch_overlap
                predicted_vol_final = predict_patchwise(model, manager, ckpt, all_input_vols, roi_location, params_dict)
                if num_outputs == 1:
                    predicted_vol_final_binarized = (predicted_vol_final[...,0] >= binarize_value[0]).astype(int)
                else:
                    predicted_vol_final_binarized = np.argmax(predicted_vol_final, axis=-1)
                #check if predicted binary map is empty; if it is, run dense prediction over the entire image just to check
                if not np.any(predicted_vol_final_binarized):
                    predicted_vol_final_binarized[:] = 1
                roi_location = find_objects(predicted_vol_final_binarized != 0)[0]
            predicted_vol_final = predict_patchwise(model, manager, ckpt, all_input_vols, roi_location, params_dict)
        else:
            output_list = predict_entire_volume(model, manager, ckpt, all_input_vols, params_dict)
            predicted_vol_final = output_list[0]
            if len(output_list) == 2:
                uncertainty_map_final = output_list[1]
        #remove any extra padding that was applied and pad back to original volume shape to ensure label maps match with ground truth
        remove_padding_initial_index = tuple([slice(start_index, predicted_vol_final.shape[i] - end_index, 1) for i, (start_index, end_index) in enumerate(pad_tuple_initial)])
        predicted_vol_final = predicted_vol_final[remove_padding_initial_index]
        pad_bounds = [(j.start, all_input_vols_orig.shape[i]-j.stop) for i,j in enumerate(volume_location)] + [(0,0)]
        predicted_vol_final = np.pad(predicted_vol_final, pad_bounds, mode='constant')
        if save_uncertainty_map[0] == True:
            uncertainty_map_final = uncertainty_map_final[remove_padding_initial_index]
            uncertainty_map_final = np.pad(uncertainty_map_final, pad_bounds, mode='constant')
            uncertainty_map_nib = nib.Nifti1Image(uncertainty_map_final[...,0], affine, header=header)
            nib.save(uncertainty_map_nib, patient_path + save_uncertainty_map[1])
        #save probability mask of each output channel
        for output_channel_predicted_vol in range(0, num_outputs):
            predicted_vol_nib = nib.Nifti1Image(predicted_vol_final[...,output_channel_predicted_vol], affine, header=header)
            if num_outputs == 1:
                #if only one output channel given, use default name
                nib.save(predicted_vol_nib, patient_path + predicted_label_name[1])
            else:
                nib.save(predicted_vol_nib, patient_path + predicted_label_name[1][:-7] + '_' + str(output_channel_predicted_vol) + '.nii.gz')
        #binarize predicted label map at requested thresholds (if only one output channel, otherwise argmax softmax)
        if num_outputs == 1:
            for threshold in binarize_value:
                predicted_vol_final_binarized = (predicted_vol_final[...,0] >= threshold).astype(int)
                #save output
                predicted_vol_nib = nib.Nifti1Image(predicted_vol_final_binarized, affine, header=header)
                if len(binarize_value) == 1:
                    #if only one threshold given, use default name
                    nib.save(predicted_vol_nib, patient_path + predicted_label_name[0])
                else:
                    nib.save(predicted_vol_nib, patient_path + 'threshold_' + str(threshold) + '_' + predicted_label_name[0])
        else:
            predicted_vol_final_binarized = np.argmax(predicted_vol_final, axis=-1)
            predicted_vol_nib = nib.Nifti1Image(predicted_vol_final_binarized, affine, header=header)
            nib.save(predicted_vol_nib, patient_path + predicted_label_name[0])
    else:
        final_probability = predict_classification(model, manager, ckpt, all_input_vols, params_dict)
        #save probability and binarized value in text files
        with open(patient_path + predicted_label_name[0], 'w') as f:
            f.write(str((final_probability > binarize_value[0]).astype(int)))
        with open(patient_path + predicted_label_name[1], 'w') as f:
            f.write(str(final_probability))

#function to predict volume patch-wise
def predict_patchwise(model, manager, ckpt, all_input_vols, roi_location, params_dict):
    patch_size, batch_size, patch_overlap, predict_fast_to_find_roi, boundary_removal, num_outputs, blend_windows_mode, number_snapshots_ensemble, percentage_empty_to_skip_patch, average_logits, logits_clip_value, predict_left_right_patch = unpack_keys(params_dict)
    batch_size = batch_size[2]
    ndims = len(patch_size)
    if predict_fast_to_find_roi[0] == True and roi_location == None:
        patch_overlap = predict_fast_to_find_roi[1]
        #number_snapshots_ensemble = [0]
    #pad volume by patch size along all dimensions
    pad_tuple_initial = tuple([(patch_size_dim // 2, patch_size_dim // 2) for patch_size_dim in patch_size] + [(0,0)])
    padded_vol_initial = np.pad(all_input_vols, pad_tuple_initial, mode='constant')
    #figure out how much to pad image given parameters for boundary removal and patch overlap
    effective_patch_size = np.array(patch_size) - (2*boundary_removal)
    patch_translation = effective_patch_size - np.ceil(effective_patch_size * patch_overlap).astype(int)
    amount_pad = patch_translation - ((np.array(padded_vol_initial.shape[:-1]) + 2*boundary_removal) % patch_translation) + (2*boundary_removal)
    amount_pad = np.array([x+patch_translation[i] if x < patch_translation[i] else x for i,x in enumerate(amount_pad)])
    amount_pad_before = np.floor(amount_pad / 2).astype(int)
    amount_pad_after = np.ceil(amount_pad / 2).astype(int)
    pad_tuple = tuple([tuple([i,j]) for (i,j) in zip(amount_pad_before, amount_pad_after)] + [(0,0)])
    padded_vol = np.pad(padded_vol_initial, pad_tuple, mode='constant')
    #pad to roi_location tuple if not None
    if roi_location != None:
        roi_location = [slice(x.start + y[0], x.stop + y[0], x.step) for x,y in zip(roi_location, pad_tuple_initial)]
        roi_location = [slice(x.start + y[0], x.stop + y[0], x.step) for x,y in zip(roi_location, pad_tuple)]
    #mirror along all requested axes
    if predict_left_right_patch[0] == True:
        flip_orientations = [i for sublist in [list(combinations(predict_left_right_patch[1], i)) for i in range(0, len(predict_left_right_patch[1])+1)] for i in sublist]
    else:
        flip_orientations = [()]
    num_mirror_combinations = len(flip_orientations)
    if batch_size <= num_mirror_combinations:
        preallocate_amount = num_mirror_combinations
    elif batch_size % num_mirror_combinations != 0:
        preallocate_amount = batch_size + num_mirror_combinations - (batch_size % num_mirror_combinations)
    else:
        preallocate_amount = batch_size
    batch = np.zeros((preallocate_amount, *patch_size, padded_vol.shape[-1]))
    batch_index = np.zeros((preallocate_amount, ndims), dtype=int)
    patchNum = 0
    loop_ranges = [range(0, padded_vol.shape[dim] - patch_size[dim], patch_translation[dim]) for dim in range(0, ndims)]
    final_loop_index = [loop_range[-1] for loop_range in loop_ranges]
    #initialize weights for blending sliding window patches together
    if blend_windows_mode[0] == 'constant':
        blend_windows_weights = np.ones(tuple([patch_size[i]-(2*boundary_removal) for i in range(0, ndims)]))
    elif blend_windows_mode[0] == 'linear':
        blend_windows_weights = np.ones([4]*ndims).astype(np.float32) * 0.1
        blend_windows_weights[tuple([slice(1,3)]*ndims)] = 1
        blend_windows_weights = scipy_zoom(blend_windows_weights, (np.array(patch_size)-(2*boundary_removal)) / 4, order=1)
    else:
        sigmas = blend_windows_mode[1]
        if not isinstance(sigmas, list):
            sigmas = [sigmas] * ndims
        #normalize sigmas to patch size
        sigmas = [sigma * (patch_size_dim-(2*boundary_removal)) for sigma, patch_size_dim in zip(sigmas, patch_size)]
        #find center value of gaussian
        center_index = [((patch_size_dim-(2*boundary_removal)) - 1.) / 2. for patch_size_dim in patch_size]
        #make (unnormalized) gaussian filter
        grid_values = np.ogrid[tuple([slice(-np.floor(center_index_dim)-1, np.ceil(center_index_dim)) for center_index_dim in center_index])]
        blend_windows_weights = np.exp(-(np.sum([grid_values_dim * grid_values_dim for grid_values_dim in grid_values])) / (2.*np.prod(sigmas)))
        #replace small values for numerical stability
        blend_windows_weights[blend_windows_weights < 0.1] = 0.1
    #patch image into batches and pass through network
    predicted_vol_list = []
    num_predicted_vol_channels = num_outputs * 2
    predicted_patch_index_boundary_removed_numpy = tuple([slice(0, None, 1)] + [slice(boundary_removal, patch_size[i]-boundary_removal, 1) for i in range(0, ndims)])
    for snapshot_number, saved_model_checkpoint in enumerate(reversed(manager.checkpoints)):
        if snapshot_number in number_snapshots_ensemble:
            if len(number_snapshots_ensemble) > 1 or snapshot_number != 0:
                ckpt.restore(saved_model_checkpoint).expect_partial()
            #pre-allocate array to hold final prediction (twice as many channels needed since we also need to hold averaging weights for each ouput channel)
            predicted_vol_snapshot = np.zeros(padded_vol.shape[:-1] + (num_predicted_vol_channels,))
            for patch_start in product(*loop_ranges):
                patch_index = tuple([slice(loop_index, loop_index + patch_size[i], 1) for i, loop_index in enumerate(patch_start)])
                patch = padded_vol[patch_index]
                #if you have used the predict_fast_to_find_roi option to find the region of interest, only run dense inference if center of patch falls within those specific bounds
                predict_this_patch = True
                if roi_location != None:
                    predict_this_patch = np.all([roi_location[i].start - predict_fast_to_find_roi[2][i] <= patch_index[i].start + patch_size[i] // 2 <= roi_location[i].stop + predict_fast_to_find_roi[2][i] if predict_fast_to_find_roi[2][i] != -1 else True for i in range(0, len(patch_size))])
                if predict_this_patch == True:
                    #only perform computation if most of center of patch is on data or if entire patch is greater than or equal to percentage_empty_to_skip_patch full
                    if (percentage_empty_to_skip_patch[1] == True and (np.mean(patch[tuple([slice(patch_size_dim//2 - 2, patch_size_dim//2 + 2) for patch_size_dim in patch_size])] != 0) >= .75)) or (np.mean(patch != 0) >= percentage_empty_to_skip_patch[0]):
                        #add this patch to batch (and optionally mirror along all requested axes)
                        for flip_axis in flip_orientations:
                            batch[patchNum,...] = np.flip(patch, axis=flip_axis)
                            batch_index[patchNum] = patch_start
                            patchNum = patchNum + 1
                #get predictions for patches once batch is filled up (or get predictions for leftover patches if it is the final loop iteration)
                if patchNum >= batch_size or (patchNum != 0 and patch_start==final_loop_index):
                    predicted_patches = model.predict(batch[:patchNum,...], batch_size=min(batch_size, patchNum))
                    #remove boundary from patch
                    predicted_patches = predicted_patches[predicted_patch_index_boundary_removed_numpy]
                    #convert into probabilities if not averaging logits
                    if average_logits == False:
                        predicted_patches = clip_logits_and_apply_activation(predicted_patches, logits_clip_value, num_outputs)
                    #add predicted patches back to predicted volume
                    for prediction_index in range(0, patchNum):
                        output_patch_index = tuple([slice(loop_index + boundary_removal, loop_index + patch_size[i] - boundary_removal, 1) for i, loop_index in enumerate(batch_index[prediction_index])])
                        predicted_vol_view = predicted_vol_snapshot[output_patch_index]
                        predicted_patch_view = predicted_patches[prediction_index]
                        #make sure to re-flip before averaging with full volume
                        predicted_patch_view = np.flip(predicted_patch_view, axis=flip_orientations[prediction_index % num_mirror_combinations])
                        for output_channel_predicted_patch, output_channel_predicted_vol in zip(range(0, num_outputs), range(0, num_predicted_vol_channels, 2)):
                            #update predicted patch with weighted average
                            predicted_vol_view[...,output_channel_predicted_vol] = np.average((predicted_vol_view[...,output_channel_predicted_vol], predicted_patch_view[...,output_channel_predicted_patch]), axis=0, weights=(predicted_vol_view[...,output_channel_predicted_vol+1], blend_windows_weights))
                            #update predicted patch with new count to ensure weighted average is correct for future patches
                            predicted_vol_view[...,output_channel_predicted_vol+1] = predicted_vol_view[...,output_channel_predicted_vol+1] + blend_windows_weights
                            #resave predicted_vol_view to full matrix
                            predicted_vol_snapshot[output_patch_index] = predicted_vol_view
                    #reset patch number
                    patchNum = 0
            predicted_vol_list.append(predicted_vol_snapshot)
    #concatenate all predicted volumes across all snapshots
    predicted_vol = np.stack(predicted_vol_list, axis=0)
    #Average over snapshots dimension to get final result
    predicted_vol = np.mean(predicted_vol, axis=0)
    #If averaging logits, need to set all un-predicted voxels (i.e. background voxels) to minimum logit so that they are assigned probability 0
    if average_logits == True:
        for output_channel_predicted_vol in range(0, num_predicted_vol_channels, 2):
            predicted_vol[...,output_channel_predicted_vol][np.where(predicted_vol[...,output_channel_predicted_vol+1] == 0)] = -logits_clip_value
        predicted_vol[...,slice(0,num_predicted_vol_channels,2)] = clip_logits_and_apply_activation(predicted_vol[...,slice(0,num_predicted_vol_channels,2)], logits_clip_value, num_outputs)
    #remove padding and return final probability masks for (cropped) volume
    remove_padding_index = tuple([slice(start_index, predicted_vol.shape[i] - end_index, 1) for i, (start_index, end_index) in enumerate(pad_tuple)])
    predicted_vol = predicted_vol[remove_padding_index][...,slice(0,num_predicted_vol_channels,2)]
    remove_padding_initial_index = tuple([slice(start_index, predicted_vol.shape[i] - end_index, 1) for i, (start_index, end_index) in enumerate(pad_tuple_initial)])
    return predicted_vol[remove_padding_initial_index]

#function to predict entire volume at once
def predict_entire_volume(model, manager, ckpt, all_input_vols, params_dict):
    patch_size, batch_size, filter_size_conv_transpose_pool, levels, atrous_conv, num_repeat_pooling, predict_multi_scale, predict_left_right_patch, number_snapshots_ensemble, num_outputs, save_uncertainty_map, average_logits, logits_clip_value = unpack_keys(params_dict)
    batch_size = batch_size[2]
    ndims = len(patch_size)
    #figure out how much to pad image to ensure that all downsampling and upsampling occurs properly
    temp_num_downsample = np.array(filter_size_conv_transpose_pool) ** (levels - 1 - atrous_conv[0])
    num_downsample = [num_downsample if repeat_pooling == -1 else pool_ratio * repeat_pooling for num_downsample, repeat_pooling, pool_ratio in zip(temp_num_downsample, num_repeat_pooling, filter_size_conv_transpose_pool)]
    #create padded inputs
    all_input_vols_list = [all_input_vols]
    if predict_multi_scale[0] == True:
        all_input_vols_list.extend([all_input_vols for _ in range(0, len(predict_multi_scale[1]))])
    pad_tuple_list = []
    padded_vol_list = []
    for i, all_input_vols_scale in enumerate(all_input_vols_list):
        if i > 0:
            all_input_vols_scale = np.pad(all_input_vols_scale, pad_tuple_list[0], mode='constant')
            all_input_vols_scale = scale_input_volume(all_input_vols_scale, predict_multi_scale[1][i-1], ndims, predict_multi_scale[2])
        #all axes must be cleanly divisible by this number
        amount_pad = np.array([0 if x % num_downsample[i] == 0 else num_downsample[i] - (x % num_downsample[i]) for i,x in enumerate(all_input_vols_scale.shape[:-1])])
        amount_pad_before = np.floor(amount_pad / 2).astype(int)
        amount_pad_after = np.ceil(amount_pad / 2).astype(int)
        pad_tuple = tuple([tuple([i,j]) for (i,j) in zip(amount_pad_before, amount_pad_after)] + [(0,0)])
        padded_vol = np.pad(all_input_vols_scale, pad_tuple, mode='constant')
        pad_tuple_list.append(pad_tuple)
        padded_vol_list.append(padded_vol)
    #create batched input to pass into the model
    if predict_left_right_patch[0] == True:
        flip_orientations = [i for sublist in [list(combinations(predict_left_right_patch[1], i)) for i in range(0, len(predict_left_right_patch[1])+1)] for i in sublist]
    else:
        flip_orientations = [()]
    num_mirror_combinations = len(flip_orientations)
    batch_list = [np.zeros((num_mirror_combinations, *padded_vol_scale.shape)) for padded_vol_scale in padded_vol_list]
    for i, padded_vol_scale in enumerate(padded_vol_list):
        for j, flip_axis in enumerate(flip_orientations):
            batch_list[i][j,...] = np.flip(padded_vol_scale, axis=flip_axis)
    #predict batch using snaphot ensemble
    predicted_vol_list = []
    for snapshot_number, saved_model_checkpoint in enumerate(reversed(manager.checkpoints)):
        if snapshot_number in number_snapshots_ensemble:
            if len(number_snapshots_ensemble) > 1 or snapshot_number != 0:
                ckpt.restore(saved_model_checkpoint).expect_partial()
            predicted_vol_snapshot = [model.predict(tf.constant(batch_scale), batch_size=batch_size) for batch_scale in batch_list]
            #un-scale predictions (if using multi-scale) and re-flip patches before averaging
            for i, predicted_scale in enumerate(predicted_vol_snapshot):
                if i > 0:
                    #un-pad scaled volumes
                    remove_padding_index = tuple([slice(0,predicted_scale.shape[0],1)] + [slice(start_index, predicted_scale.shape[k+1] - end_index, 1) for k, (start_index, end_index) in enumerate(pad_tuple_list[i])])
                    predicted_scale = predicted_scale[remove_padding_index]
                    #un-scale volumes
                    predicted_scale = np.stack([scale_input_volume(predicted_scale[j,...], 1 / predict_multi_scale[1][i-1], ndims, predict_multi_scale[2]) for j in range(0, predicted_scale.shape[0])], axis=0)
                #re-flip patches
                for j, flip_axis in enumerate(flip_orientations):
                    predicted_scale[j,...] = np.flip(predicted_scale[j,...], axis=flip_axis)
                predicted_vol_snapshot[i] = predicted_scale
            #concatenate all multi-scale and flipped predictions together for this snapshot
            predicted_vol_snapshot = np.concatenate(predicted_vol_snapshot, axis=0)
            predicted_vol_list.append(predicted_vol_snapshot)
    #concatenate all predicted volumes across all snapshots
    predicted_vol = np.concatenate(predicted_vol_list, axis=0)
    #if making uncertainty maps, use mean entropy over samples
    if save_uncertainty_map[0] == True:
        probability_map = clip_logits_and_apply_activation(np.copy(predicted_vol), logits_clip_value, num_outputs)
        if num_outputs == 1:
            uncertainty_map = np.mean(-(probability_map * np.log2(probability_map) + (1 - probability_map) * np.log2(1 - probability_map)), axis=0)
        else:
            uncertainty_map = np.expand_dims(np.mean(np.sum(-probability_map * np.log2(probability_map), axis=-1), axis=0), axis=-1)
    #either convert to probability and then average, or average logits and then convert to probabilities depending on parameter setting
    if average_logits == False:
        predicted_vol = clip_logits_and_apply_activation(predicted_vol, logits_clip_value, num_outputs)
        predicted_vol = np.mean(predicted_vol, axis=0)
    else:
        predicted_vol = np.mean(predicted_vol, axis=0)
        predicted_vol = clip_logits_and_apply_activation(predicted_vol, logits_clip_value, num_outputs)
    #remove padding and return final probability masks for (cropped) volume
    remove_padding_index = tuple([slice(start_index, predicted_vol.shape[i] - end_index, 1) for i, (start_index, end_index) in enumerate(pad_tuple_list[0])])
    if save_uncertainty_map[0] == True:
        return [predicted_vol[remove_padding_index], uncertainty_map[remove_padding_index]]
    else:
        return [predicted_vol[remove_padding_index]]

#function to predict binary output from classification network
def predict_classification(model, manager, ckpt, all_input_vols, params_dict):
    number_snapshots_ensemble, batch_size, average_logits, logits_clip_value, num_outputs = unpack_keys(params_dict)
    batch_size = batch_size[2]
    all_input_vols_batch = make_batch_of_crops(all_input_vols, params_dict)
    model_predictions = np.zeros((len(number_snapshots_ensemble), all_input_vols_batch.shape[0]))
    counter = 0
    for snapshot_number, saved_model_checkpoint in enumerate(reversed(manager.checkpoints)):
        if snapshot_number in number_snapshots_ensemble:
            if len(number_snapshots_ensemble) > 1 or snapshot_number != 0:
                ckpt.restore(saved_model_checkpoint).expect_partial()
            model_predictions[counter, :] = np.squeeze(model.predict(all_input_vols_batch, batch_size=batch_size))
            counter = counter + 1
    if average_logits == True:
        final_averaged_logit = np.array(np.mean(model_predictions))
        final_probability = clip_logits_and_apply_activation(final_averaged_logit, logits_clip_value, num_outputs)
    else:
        final_probability = clip_logits_and_apply_activation(model_predictions, logits_clip_value, num_outputs)
        final_probability = np.mean(final_probability)
    return final_probability

#helper function to generate desired number of crops from image
def make_batch_of_crops(all_input_vols, params_dict):
    patch_size, patch_overlap, boundary_removal = unpack_keys(params_dict)
    #get stride in every direction using patch size and patch overlap
    stride = [np.floor(max(1, (patch_size_dim - 2 * boundary_removal) * (1-patch_overlap))).astype(int) for patch_size_dim in patch_size]
    batch = []
    #make loop ranges and then account for boundary removal if applicable
    loop_ranges = [range(0, max(1, volume_shape_dim - patch_size_dim), stride_dim) for volume_shape_dim, patch_size_dim, stride_dim in zip(all_input_vols.shape[:-1], patch_size, stride)]
    loop_ranges = [range(loop_range.start + boundary_removal, loop_range.stop - boundary_removal, loop_range.step) if (loop_range.stop - loop_range.start > 2 * boundary_removal) else loop_range for loop_range in loop_ranges]
    for patch_start in product(*loop_ranges):
        patch_index = tuple([slice(loop_index, loop_index + patch_size[i], 1) for i, loop_index in enumerate(patch_start)] + [slice(None,None)])
        batch.append(all_input_vols[patch_index])
    batch = np.stack(batch, axis=0)
    return batch

#helper function to load volumes from list
def load_volumes_nibabel(patient_dir, volume_name_list):
    for j, name in enumerate(volume_name_list):
        nib_vol = nib.load(patient_dir + name)
        image = np.round(nib_vol.get_data().astype(np.float32), 5)
        if j == 0:
            affine = nib_vol.get_affine()
            header = nib_vol.get_header()
            all_volumes = np.zeros((image.shape) + ((len(volume_name_list),)))
        all_volumes[...,j] = image
    return all_volumes, affine, header

#helper function to clip logits and get sigmoid or softmax probability
def clip_logits_and_apply_activation(logits_array, logits_clip_value, num_outputs):
    #clip logit values to prevent overflow errors in activation function
    np.clip(logits_array, -logits_clip_value, logits_clip_value, out=logits_array)
    if num_outputs == 1:
        #compute sigmoid probability
        return 1 / (1 + np.exp(-logits_array))
    else:
        #compute softmax probability
        exp_array = np.exp(logits_array)
        return exp_array / np.expand_dims(np.sum(exp_array, axis=-1), axis=-1)

#helper function to scale input volume
def scale_input_volume(all_input_vols_scale, scale_factor, ndims, order):
    #pre-allocated scaled array
    scaled_vol_size = [int(all_input_vols_scale.shape[i] * scale_factor) for i in range(0, ndims)] + [all_input_vols_scale.shape[-1]]
    scaled_volume = np.zeros(scaled_vol_size)
    #create meshgrid that will be used for scaling
    meshgrid_bounds = [tuple([0, all_input_vols_scale.shape[i], 1/scale_factor]) for i in range(0, ndims)]
    range_tuple = [np.arange(*meshgrid_bounds[i]) for i in range(0, ndims)]
    meshgrid_array = np.meshgrid(*range_tuple, indexing='ij')
    #map_coordinates to compute scaling transformation
    for i in range(0, all_input_vols_scale.shape[-1]):
        scaled_volume[...,i] = map_coordinates(all_input_vols_scale[...,i], meshgrid_array, order=order, output=np.float32)
    return scaled_volume