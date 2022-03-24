#TODO: load_volumes helper function is messy due to handling of potentially missing inputs -> could be rewritten to be cleaner at some point
#TODO: Make sure that during channel dropout, always have one real input (if it exists)
#TODO: Use np.copy to make copies of volume/label to prevent issues with array views if they exist
#TODO: Future augmentations to add: Intensity - saturation, contrast, brightness
#TODO: gamma correction factor sampling (i.e. better to darken or brighten image)?
#TODO: cache nonzero values, min, max of input channels to marginally speed up gamma correction
#TODO: why does data loading take longer for large CT volumes than it does for brain volumes
#TODO: ensure that data loading for classification network works as expected for multi-class problems
#TODO: on_epoch_end() is called twice, which is messing up the augmentation scheduler and epoch counter (https://github.com/tensorflow/tensorflow/issues/35911 / https://github.com/tensorflow/tensorflow/issues/43184)

import os
import random
import numpy as np
import nibabel as nib
import tensorflow as tf

from copy import deepcopy
from sorcery import unpack_keys
from scipy.ndimage import find_objects
from scipy.ndimage import center_of_mass
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from scipy.spatial.transform import Rotation
from scipy.ndimage import zoom as scipy_zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label as scipy_conn_comp
from scipy.ndimage import generate_binary_structure
from scipy.ndimage.interpolation import map_coordinates

#Data generator class to use with model.fit
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, patients, batch_size, num_patches_per_patient, full_image_patching, name, params_dict):
        #unpack relevant dictionary elements
        input_image_names, ground_truth_label_names, input_segmentation_masks, patch_size, sample_background_prob, run_augmentations_every_epoch, voxel_spacing, downsample_for_augmentation, run_augmentations_in_2D, crop_input_image_for_augmentation, probability_augmentation, augmentation_scheduler, gamma_correction, deform_mesh, scale_mesh, rotate_mesh, shear_mesh, translate_mesh, left_right_flip, erode_dilate_conn_comp, randomize_order_augmentations, spline_order, num_outputs, use_conn_comp_patching, patch_inside_volume, require_positive_class_per_batch, balanced_batches_for_classification, balanced_epochs_for_classification, patch_center_of_tumor_for_classification, cache_volumes, filter_size_conv_transpose_pool, num_repeat_pooling, levels, atrous_conv, network_name, output_file = unpack_keys(params_dict)
        self.data_dir = data_dir
        self.patients = patients
        self.batch_size = batch_size
        self.num_patches_per_patient = num_patches_per_patient
        self.name = name
        self.input_image_names = input_image_names
        self.ground_truth_label_names = ground_truth_label_names
        self.input_segmentation_masks = deepcopy(input_segmentation_masks)
        if not isinstance(self.input_segmentation_masks, list):
            self.input_segmentation_masks = [self.input_segmentation_masks] * len(self.input_image_names)
        self.patch_size = patch_size
        self.ndims = len(self.patch_size)
        self.sample_background_prob = sample_background_prob
        self.run_augmentations_every_epoch = deepcopy(run_augmentations_every_epoch)
        self.voxel_spacing = deepcopy(voxel_spacing)
        if downsample_for_augmentation[0] == True:
            self.voxel_spacing[1] = voxel_spacing[1] / downsample_for_augmentation[1]
            self.voxel_spacing[0] = False
        self.run_augmentations_in_2D = run_augmentations_in_2D
        self.crop_input_image_for_augmentation = deepcopy(crop_input_image_for_augmentation)
        if not isinstance(self.crop_input_image_for_augmentation[1], list):
            self.crop_input_image_for_augmentation[1] = [self.crop_input_image_for_augmentation[1]] * self.ndims
        if self.run_augmentations_in_2D == True:
            self.voxel_spacing[1][-1] = 1
            self.crop_input_image_for_augmentation[1][-1] = 0
        if self.crop_input_image_for_augmentation[0] == True and self.crop_input_image_for_augmentation[2] == False:
            self.voxel_spacing[0] = True
        self.gamma_correction = deepcopy(gamma_correction)
        self.deform_mesh = deepcopy(deform_mesh)
        self.scale_mesh = deepcopy(scale_mesh)
        self.rotate_mesh = deepcopy(rotate_mesh)
        self.shear_mesh = deepcopy(shear_mesh)
        self.translate_mesh = deepcopy(translate_mesh)
        self.left_right_flip = deepcopy(left_right_flip)
        self.erode_dilate_conn_comp = deepcopy(erode_dilate_conn_comp)
        self.randomize_order_augmentations = randomize_order_augmentations
        self.spline_order = deepcopy(spline_order)
        if len(self.spline_order) == 2:
            self.spline_order = self.spline_order + [self.spline_order[0]]
        self.num_outputs = np.maximum(num_outputs,2)
        self.use_conn_comp_patching = use_conn_comp_patching
        self.patch_inside_volume = patch_inside_volume
        self.require_positive_class_per_batch = require_positive_class_per_batch
        self.balanced_batches_for_classification = balanced_batches_for_classification
        self.balanced_epochs_for_classification = balanced_epochs_for_classification
        self.patch_center_of_tumor_for_classification = patch_center_of_tumor_for_classification
        self.cache_volumes = cache_volumes
        self.full_image_patching = full_image_patching
        self.filter_size_conv_transpose_pool = filter_size_conv_transpose_pool
        self.levels = levels
        self.atrous_conv = atrous_conv
        self.network_name = network_name
        #if using full image patching, figure out how much to pad image to ensure that all downsampling and upsampling occurs properly
        temp_num_downsample = np.array(self.filter_size_conv_transpose_pool) ** (self.levels - 1 - self.atrous_conv[0])
        self.num_downsample = [num_downsample if repeat_pooling == -1 else pool_ratio * repeat_pooling for num_downsample, repeat_pooling, pool_ratio in zip(temp_num_downsample, num_repeat_pooling, filter_size_conv_transpose_pool)]
        self.output_file = output_file
        self.probability_augmentation = deepcopy(probability_augmentation)
        self.probability_augmentation = self.probability_augmentation + [prob_augment[1] for prob_augment in self.erode_dilate_conn_comp]
        num_augmentations = len(self.probability_augmentation)
        if not isinstance(self.probability_augmentation, list):
            self.probability_augmentation = [self.probability_augmentation] * num_augmentations
        self.augment_data = [True, self.probability_augmentation]
        self.augmentation_scheduler = augmentation_scheduler
        if self.augmentation_scheduler[0] == True:
            self.final_augment_prob = np.array(self.augment_data[1])
            self.increment_per_epoch = self.final_augment_prob / self.augmentation_scheduler[1]
            self.augment_data[1] = [0.0] * num_augmentations
        #ensure number of patients per epoch cleanly divides batch size
        last_batch_size = (len(self.patients)*self.num_patches_per_patient) % self.batch_size
        if last_batch_size == 0:
            pad_samples = 0
        else:
            pad_samples = self.batch_size - last_batch_size
        self.pad_samples = pad_samples
        self.patients_list_all = list(np.repeat(self.patients, self.num_patches_per_patient)) + self.patients[0:self.pad_samples]
        self.num_samples_per_epoch = len(self.patients_list_all)
        self.num_batches_per_epoch = int(np.ceil(self.num_samples_per_epoch / self.batch_size))
        #only shuffle and augment training set
        if self.name != 'train':
            self.augment_data[0] = False
            self.left_right_flip[0] = False
            self.gamma_correction[0] = False
            self.deform_mesh[0] = False
            self.scale_mesh[0] = False
            self.rotate_mesh[0] = False
            self.shear_mesh[0] = False
            self.translate_mesh[0] = False
            self.erode_dilate_conn_comp[0][0] = False
            self.erode_dilate_conn_comp[1][0] = False
            self.erode_dilate_conn_comp[2][0] = False
            self.run_augmentations_every_epoch = [True, 1]  #always batch fresh if validation data (to ensure proper translation augmentation)
        else:
            random.shuffle(self.patients_list_all)
            if self.network_name != 'Unet' and (self.balanced_batches_for_classification == True or self.balanced_epochs_for_classification[0] == True):
                temp_classification_labels_list = np.zeros(len(self.patients))
                for i, patient in enumerate(self.patients):
                    temp_classification_labels_list[i] = self.load_classification(patient, self.ground_truth_label_names)
                self.classification_labels_array = np.zeros((len(temp_classification_labels_list), 2))
                self.classification_labels_array[:,0] = temp_classification_labels_list
                self.classification_labels_array[:,1] = range(0, len(temp_classification_labels_list))
                self.balance_classification_dataset()
        self.perform_augmentation = ~np.array([self.deform_mesh[0], self.scale_mesh[0], self.rotate_mesh[0], self.shear_mesh[0], self.translate_mesh[0]])
        self.input_patch_shape = (*self.patch_size, len(self.input_image_names))
        if self.network_name == 'Unet':
            self.output_patch_shape = (*self.patch_size, len(self.ground_truth_label_names))
        else:
            self.output_patch_shape = (len(self.ground_truth_label_names),)
        self.patch_regions = [int(x / 2) for x in self.patch_size]
        self.structuring_element = generate_binary_structure(self.ndims, self.ndims)
        #cache data is requested
        if self.cache_volumes == True:
            if self.run_augmentations_every_epoch[0] == False:
                self.augmented_data_cache = {}
            self.data_cache = {}
            self.cache_patients_into_memory()
        else:
            self.run_augmentations_every_epoch = [True, 1]
        #epoch counter
        self.epoch = 0
        #print some information about batches to text file
        with open(self.output_file, 'a') as f:
            f.write(str(self.num_samples_per_epoch) + ' samples (' + str(self.num_batches_per_epoch) + ' batches) per epoch in ' + self.name + ' dataset \n')
    
    #Number of batches per epoch
    def __len__(self):
        return self.num_batches_per_epoch
    
    #generates one batch of data
    def __getitem__(self, index):
        #generate indexes of the batch
        if index < self.num_batches_per_epoch - 1:
            batch_indexes = slice(index*self.batch_size, (index+1)*self.batch_size, 1)
        else:
            batch_indexes = slice(index*self.batch_size, self.num_samples_per_epoch, 1)
        #generate batch volumes/labels
        x_batch, y_batch = self.generate_patches(self.patients_list_all[batch_indexes])
        #if last index in epoch, increment epoch counter
        if index == self.num_batches_per_epoch - 1:
            self.epoch = self.epoch + 1
        return (x_batch, y_batch)
    
    #shuffles (training) data at the end of every epoch; also applies augmentation scheduling if desired
    def on_epoch_end(self):
        if self.augment_data[0] == True:
            #re-make patients list at end of epoch in case we needed to pad list to make training batches equal (this ensures that new cases are "duplicated" each epoch for fairness)
            self.patients_list_all = list(np.repeat(self.patients, self.num_patches_per_patient)) + random.sample(self.patients, self.pad_samples)
            random.shuffle(self.patients_list_all)
            if self.network_name != 'Unet' and (self.balanced_batches_for_classification == True or self.balanced_epochs_for_classification[0] == True):
                self.balance_classification_dataset()
            if self.augmentation_scheduler[0] == True:
                self.augment_data[1] = np.min([self.augment_data[1] + self.increment_per_epoch, self.final_augment_prob], axis=0)
            #shuffle cached augmentations (if they exists)
            if self.run_augmentations_every_epoch[0] == False:
                if (self.epoch % self.run_augmentations_every_epoch[1]) == 0:
                    for patient in self.patients:
                        self.augmented_data_cache[patient] = []
                else:
                    [random.shuffle(self.augmented_data_cache[patient]) for patient in self.patients]
    
    #generates patches for all patients in dataset
    def generate_patches(self, patients_list_batch):
        len_patients_list_batch = len(patients_list_batch)
        x_batch, y_batch = np.zeros((len_patients_list_batch, *self.input_patch_shape)), np.zeros((len_patients_list_batch, *self.output_patch_shape))
        for counter, patient in enumerate(patients_list_batch):
            #load cropped volumes from data cache or from memory directly
            if self.cache_volumes == True:
                #if running augmentations every epoch or if at epoch when we need to re-cache data
                if self.run_augmentations_every_epoch[0] == True or (self.epoch % self.run_augmentations_every_epoch[1]) == 0:
                    cropped_image, cropped_label, cropped_nonzero_image_channel, cropped_nonzero_label_channel = self.data_cache[patient]
                #grab augmented data directly from saved augmented data cache and ignore everything else below
                else:
                    #extract first image/label and move to end of list
                    augmented_patch_label = self.augmented_data_cache[patient].pop(0)
                    self.augmented_data_cache[patient].append(augmented_patch_label)
                    x_batch[counter,...], y_batch[counter,...] = augmented_patch_label[0], augmented_patch_label[1]
                    continue
            else:
                cropped_image, cropped_label, cropped_nonzero_image_channel, cropped_nonzero_label_channel = self.load_and_crop(patient)
            #randomly choose input volumes if swapping multiple inputs into the same channel (i.e. T2 and FLAIR randomly chosen to put into the same channel)
            cropped_image, cropped_nonzero_image_channel = self.randomly_choose_inputs(cropped_image, cropped_nonzero_image_channel, self.input_image_names)
            if self.network_name == 'Unet':
                cropped_label, cropped_nonzero_label_channel = self.randomly_choose_inputs(cropped_label, cropped_nonzero_label_channel, self.ground_truth_label_names)
            #if using full volumes, no need to generate patches:
            if self.full_image_patching == True:
                #pad all axes to the patch size in case cropped volume is smaller than size of patch along an axis or axes
                extra_padding = np.maximum(self.patch_size - np.array(cropped_image.shape[:-1]), 0)
                pad_tuple_initial = tuple([(int(np.floor(i / 2)), int(np.ceil(i / 2))) for i in extra_padding] + [(0,0)])
                cropped_image = np.pad(cropped_image, pad_tuple_initial, mode='constant')
                if self.network_name == 'Unet':
                    cropped_label = np.pad(cropped_label, pad_tuple_initial, mode='constant')
                #all axes must be cleanly divisible to ensure no upsampling/downsampling issues
                amount_pad = np.array([0 if x % self.num_downsample[i] == 0 else self.num_downsample[i] - (x % self.num_downsample[i]) for i,x in enumerate(cropped_image.shape[:-1])])
                amount_pad_before = np.floor(amount_pad / 2).astype(int)
                amount_pad_after = np.ceil(amount_pad / 2).astype(int)
                pad_tuple = tuple([tuple([i,j]) for (i,j) in zip(amount_pad_before, amount_pad_after)] + [(0,0)])
                cropped_image = np.pad(cropped_image, pad_tuple, mode='constant')
                if self.network_name == 'Unet':
                    cropped_label = np.pad(cropped_label, pad_tuple, mode='constant')
                #pad current x_batch/y_batch or input image/label so that all axes match for stacking
                #first pad x_batch/y_batch
                amount_pad = np.array([max(x - x_batch.shape[i+1], 0) for i,x in enumerate(cropped_image.shape[:-1])])
                amount_pad_before = np.floor(amount_pad / 2).astype(int)
                amount_pad_after = np.ceil(amount_pad / 2).astype(int)
                pad_tuple = tuple([(0,0)] + [tuple([i,j]) for (i,j) in zip(amount_pad_before, amount_pad_after)] + [(0,0)])
                x_batch = np.pad(x_batch, pad_tuple, mode='constant')
                if self.network_name == 'Unet':
                    y_batch = np.pad(y_batch, pad_tuple, mode='constant')
                #now pad image/label
                amount_pad = np.array([max(x_batch.shape[i+1] - x, 0) for i,x in enumerate(cropped_image.shape[:-1])])
                amount_pad_before = np.floor(amount_pad / 2).astype(int)
                amount_pad_after = np.ceil(amount_pad / 2).astype(int)
                pad_tuple = tuple([tuple([i,j]) for (i,j) in zip(amount_pad_before, amount_pad_after)] + [(0,0)])
                cropped_image = np.pad(cropped_image, pad_tuple, mode='constant')
                if self.network_name == 'Unet':
                    cropped_label = np.pad(cropped_label, pad_tuple, mode='constant')
            if self.network_name == 'Unet':
                if self.use_conn_comp_patching == True:
                    #patching via connected components (works well when have many distinct lesions)
                    conn_comp = scipy_conn_comp(cropped_label[...,0], self.structuring_element)
                else:
                    #use sparse (non one-hot encoded) ground truth label to identify patching regions
                    conn_comp = (cropped_label[...,0], self.num_outputs-1)
                #choose one lesion (or background) at random to patch over
                #all lesions given equal weight currently (with background given constant probability = sample_background_prob)
                if conn_comp[1] > 0:
                    prob = np.repeat((1 - self.sample_background_prob) / conn_comp[1], conn_comp[1] + 1)
                else:
                    prob = np.array([self.sample_background_prob])
                prob[0] = self.sample_background_prob
                #ensure probabilities sum to 1
                prob = prob / np.sum(prob)
                #optionally force last patch in batch to have positive class example if no positive class examples exist yet (which will help smooth training, especially at small batch sizes)
                if self.require_positive_class_per_batch == True and counter == len_patients_list_batch-1:
                    #check if positive class present in previous batches
                    if not np.any(y_batch[:counter,...]):
                        #remove background as choice for random sampling to force positive class example during
                        if cropped_nonzero_label_channel[0] == True and len(prob) > 1:
                            prob[0] = 0.
                            prob = prob / np.sum(prob)
            else:
                conn_comp = None
                prob = None
            #generate patch from patient
            x_batch[counter,...], y_batch[counter,...] = self.patch_patient_numpy(cropped_image, cropped_label, cropped_nonzero_image_channel, conn_comp, prob)
            #add augmented patch to augmented data cache if correct epoch to do so
            if self.run_augmentations_every_epoch[0] == False and (self.epoch % self.run_augmentations_every_epoch[1]) == 0:
                self.augmented_data_cache[patient].append([x_batch[counter,...], y_batch[counter,...]])
        return x_batch, y_batch

    #patches patient into desired number of patches
    def patch_patient_numpy(self, cropped_image, cropped_label, cropped_nonzero_image_channel, conn_comp, prob):
        #shape of input image (not including channel dimension)
        cropped_image_shape = cropped_image.shape[:-1]
        if self.network_name == 'Unet':
            #for some multi-class segmentation tasks, not all labels present in each image so make sure region you are picking exists
            found_lesion = False
            while found_lesion == False:
                #choose which lesion (or background class) to sample from
                lesion_num = np.random.choice(conn_comp[1]+1, p=prob)
                #choose random lesion (or background) to make patch (if background, ensure that center voxel is not outside skull stripped area)
                masked_label = conn_comp[0] == lesion_num
                if np.any(masked_label == True):
                    found_lesion = True
            if lesion_num != 0:
                lesion_loc = np.where(masked_label == 1)
            else:
                #only need to multiply masked label with first input volume (since all are skull stripped similarly)
                background_brain_region = np.multiply(masked_label, cropped_image[...,0])
                lesion_loc = np.where(background_brain_region != 0)
            #choose random voxel in chosen lesion (or background) as the center of the patch
            temp_index = np.random.choice(len(lesion_loc[0]))
            center_index = [x[temp_index] for x in lesion_loc]
            #if this is a background patch and have set flag to only patch inside the volume, re-center patch (or if using full-sized images)
            if (lesion_num == 0 and self.patch_inside_volume == True) or self.full_image_patching == True:
                center_index = self.get_center_index_from_volume_shape(cropped_image_shape)
        else:
            if self.patch_center_of_tumor_for_classification == False:
                center_index = self.get_center_index_from_volume_shape(cropped_image_shape)
            else:
                center_index = np.array(center_of_mass(cropped_image[...,-1])).astype(int)
        #perform gamma correction on cropped volume if requested to do so (first augmentation so that zero values outside skull-stripped region can be ignored)
        if self.gamma_correction[0] == True:
            cropped_image = self.gamma_correct_volume(cropped_image, cropped_nonzero_image_channel)
        #augment label maps if passing binary masks as input
        if np.any(self.input_segmentation_masks):
            cropped_image = self.erode_dilate_connected_components_input_label_map(cropped_image, cropped_nonzero_image_channel)
        #augment and patch image and label
        if self.full_image_patching == True:
            patient_input_patch_shape = (*cropped_image_shape, len(self.input_image_names))
            patient_output_patch_shape = (*cropped_image_shape, len(self.ground_truth_label_names))
            patched_image, patched_label = np.zeros(patient_input_patch_shape), np.zeros(patient_output_patch_shape)
        else:
            patched_image, patched_label = np.zeros(self.input_patch_shape), np.zeros(self.output_patch_shape)
        do_augmentation = np.ma.masked_array(np.random.random(len(self.perform_augmentation)), mask=self.perform_augmentation) < self.augment_data[1][2:2+len(self.perform_augmentation)]
        if np.any(do_augmentation) == True:
            if self.network_name == 'Unet':
                cropped_label_view = np.expand_dims(cropped_label[...,0], axis=-1)
            else:
                cropped_label_view = np.expand_dims(cropped_image[...,-1], axis=-1)
            if self.voxel_spacing[0] == False:
                #non-isotropic data so resample cropped label view (only if elastic deformation being applied to save time)
                if do_augmentation[0] == True and self.network_name == 'Unet':
                    cropped_label_view = np.expand_dims(scipy_zoom(cropped_label_view[...,0], self.voxel_spacing[1], order=self.spline_order[1]), axis=-1)
                    cropped_image_shape = cropped_label_view.shape[:-1]
                else:
                    cropped_image_shape = tuple([int(round(ii * jj)) for ii, jj in zip(cropped_image_shape, self.voxel_spacing[1])])
                #transform center index to fit new volume resolution
                center_index = (center_index * self.voxel_spacing[1]).astype(int)
            #find how much to pad mesh grid by to ensure no early clipping of the final patch
            patch_bounds, pad_bounds = self.get_patch_and_pad_bounds(center_index, cropped_image_shape, correct_patch_regions=True)
            #create meshgrid over entire volume area
            meshgrid_bounds = [tuple([-pad_bounds[i][0], cropped_image_shape[i] + pad_bounds[i][1]]) for i in range(0, self.ndims)]
            if self.run_augmentations_in_2D == True:
                #create meshgrid over all slices selected for this patch (ignores slices not in the patch since augmentations wont be affected by those slices)
                meshgrid_bounds[-1] = (patch_bounds[-1].start - pad_bounds[-2][0], patch_bounds[-1].stop + pad_bounds[-2][1])
            #center meshgrid on center of mass of volume (this will ensure proper affine transforms (scaling, rotation, shearing) is applied)
            meshgrid_bounds = [(meshgrid_bound[0] - center_index_dim, meshgrid_bound[1] - center_index_dim) for meshgrid_bound, center_index_dim in zip(meshgrid_bounds, center_index)]
            #crop meshgrid bounds if need to speed up augmentations
            padding_amounts_for_meshgrid = None
            if self.crop_input_image_for_augmentation[0] == True:
                if self.full_image_patching == True:
                    current_patch_regions = [int(x / 2) for x in cropped_image_shape]
                else:
                    current_patch_regions = self.patch_regions
                    if self.voxel_spacing[0] == False:
                        current_patch_regions = (current_patch_regions * self.voxel_spacing[1]).astype(int)
                old_meshgrid_bounds = deepcopy(meshgrid_bounds)
                meshgrid_bounds = [(max(meshgrid_bound[0], -patch_region-voxel_boundary), min(meshgrid_bound[1], patch_region+voxel_boundary)) for meshgrid_bound, patch_region, voxel_boundary in zip(old_meshgrid_bounds, current_patch_regions, self.crop_input_image_for_augmentation[1])]
                padding_amounts_for_meshgrid = [(np.abs(meshgrid_bound[0] - old_meshgrid_bound[0]), np.abs(meshgrid_bound[1] - old_meshgrid_bound[1])) for meshgrid_bound, old_meshgrid_bound in zip(meshgrid_bounds, old_meshgrid_bounds)] + [(0,0)]
            #create mesh grid that will hold transforms large enough to not require any additional padding at end
            range_tuple = [np.arange(*meshgrid_bounds[i]) for i in range(0, self.ndims)]
            meshgrid_array = np.meshgrid(*range_tuple, indexing='ij')
            meshgrid_shape = meshgrid_array[0].shape
            #flatten meshgrid and transpose so that channel dimension is last
            meshgrid_array = np.array([np.reshape(meshgrid_array[i], (-1,)) for i in range(0, self.ndims)]).transpose(1,0)
            #run spatial transforms on meshgrid if requested to do so
            translate_factors = np.zeros(self.ndims)
            if do_augmentation[0] == True:
                #apply elastic deformation
                meshgrid_array = self.elastic_deform_mesh_grid(meshgrid_array, cropped_label_view, patch_bounds, pad_bounds, padding_amounts_for_meshgrid=padding_amounts_for_meshgrid, cropped_image_shape=cropped_image_shape)
            if np.any(do_augmentation[1:4]):
                #apply affine transforms (scale, rotate, shear)
                meshgrid_array = self.compose_transforms_mesh_grid(meshgrid_array, do_augmentation)
            if do_augmentation[4] == True:
                #find translation factors
                translate_factors = np.random.uniform(*self.translate_mesh[1], self.ndims)
                if self.ndims == 3 and self.run_augmentations_in_2D == True:
                    translate_factors[-1] = 0
            #uncenter meshgrid
            meshgrid_array = meshgrid_array + (center_index + translate_factors)
            #transform center index to new location
            center_index_transformed = [center_index[i] + pad_bounds[i][0] + translate_factors[i] for i in range(0, self.ndims)]
            #reshape mesh grid back into size of cropped image and transpose so that channel dimension is last
            meshgrid_array = np.array([np.reshape(meshgrid_array[:,i], meshgrid_shape) for i in range(0, self.ndims)]).transpose(*range(1,self.ndims+1), 0)
            #if non-isotropic data, resample back to original resolution
            if self.voxel_spacing[0] == False:
                #revert to the original volume size
                scipy_zoomed_cropped_image_shape = np.copy(cropped_image_shape)
                cropped_image_shape = cropped_image.shape[:-1]
                factor_scale_meshgrid = np.array(cropped_image_shape) / np.array(scipy_zoomed_cropped_image_shape)
                #transform center index back to original volume resolution
                center_index_transformed = center_index_transformed * factor_scale_meshgrid
                #resample meshgrid back to original space
                meshgrid_array_isotropic = np.copy(meshgrid_array)
                meshgrid_shape = tuple([int(round(meshgrid_shape_dim * factor_scale_meshgrid_dim)) for meshgrid_shape_dim, factor_scale_meshgrid_dim in zip(meshgrid_shape, factor_scale_meshgrid)])
                meshgrid_array = np.zeros((*meshgrid_shape, self.ndims))
                for j in range(0, meshgrid_array.shape[-1]):
                    meshgrid_array[...,j] = scipy_zoom(meshgrid_array_isotropic[...,j], factor_scale_meshgrid, order=1)
                    meshgrid_array[...,j] = meshgrid_array[...,j] * factor_scale_meshgrid[j]
            #convert to integer so that we can use as index in matrix
            center_index_transformed = np.round(center_index_transformed).astype(int)
            #pad meshgrid back to original size in case we had cropped it for speed
            if self.crop_input_image_for_augmentation[0] == True:
                meshgrid_array = np.pad(meshgrid_array, padding_amounts_for_meshgrid, mode='constant')
                meshgrid_shape = meshgrid_array[...,0].shape
            #patch and transpose meshgrid
            patch_bounds, pad_bounds = self.get_patch_and_pad_bounds(center_index_transformed, meshgrid_shape, true_patch_regions=cropped_image_shape)
            if self.run_augmentations_in_2D == True:
                list_patch_bounds = list(patch_bounds)
                list_patch_bounds[-1] = slice(None, None, 1)
                patch_bounds = tuple(list_patch_bounds)
            patched_meshgrid_array = meshgrid_array[patch_bounds].transpose(-1, *range(0,self.ndims))
            #pre-filtering large image can slow down map_coordinates function -> crop image based on min/max values in patched_meshgrid_array
            final_patch_bounds = []
            voxel_boundary = 5
            for j in range(0, self.ndims):
                if self.full_image_patching == True:
                    min_index_dim = None
                    max_index_dim = None
                else:
                    min_index_dim = int(max(np.min(patched_meshgrid_array[j,...]) - voxel_boundary, 0))
                    max_index_dim = int(min(np.max(patched_meshgrid_array[j,...]) + voxel_boundary, cropped_image_shape[j]))
                    patched_meshgrid_array[j,...] = patched_meshgrid_array[j,...] - min_index_dim
                final_patch_bounds.append(slice(min_index_dim, max_index_dim))
            final_patch_bounds = tuple(final_patch_bounds)
            #apply transforms to padded image and label to acquire final patch
            pad_bounds = pad_bounds[:-1]
            if self.run_augmentations_in_2D == True:
                #do not need to pad z-dimension since already proper size if using 2D augmentations
                pad_bounds = pad_bounds[:-1] + ((0,0),)
            for j in range(0, patched_image.shape[-1]):
                if self.input_segmentation_masks[j] == False:
                    label_spline_order = self.spline_order[0]
                else:
                    label_spline_order = self.spline_order[1]
                transformed_image = map_coordinates(cropped_image[final_patch_bounds + (j,)], patched_meshgrid_array, order=label_spline_order, output=np.float32)
                patched_image[...,j] = np.pad(transformed_image, pad_bounds, mode='constant')
            if self.network_name == 'Unet':
                for j in range(0, patched_label.shape[-1]):
                    if j == 0:
                        label_spline_order = self.spline_order[1]
                    else:
                        label_spline_order = self.spline_order[2]
                    transformed_label = map_coordinates(cropped_label[final_patch_bounds + (j,)], patched_meshgrid_array, order=label_spline_order, output=np.float32)
                    patched_label[...,j] = np.pad(transformed_label, pad_bounds, mode='constant')
        else:
            patch_bounds, pad_bounds = self.get_patch_and_pad_bounds(center_index, cropped_image_shape)
            patched_image = np.pad(cropped_image[patch_bounds], pad_bounds, mode='constant')
            if self.network_name == 'Unet':
                patched_label = np.pad(cropped_label[patch_bounds], pad_bounds, mode='constant')
        #randomly mirror patched volumes if requested to do so
        if self.left_right_flip[0] == True:
            patched_image, patched_label = self.mirror_patch(patched_image, patched_label)
        if self.network_name == 'Unet':
            return patched_image, patched_label
        else:
            return patched_image, cropped_label
    
    #function to elastically deform mesh grid
    def elastic_deform_mesh_grid(self, meshgrid_array, cropped_label, patch_bounds, pad_bounds, padding_amounts_for_meshgrid=None, cropped_image_shape=None):
        #binarize sparse (non one-hot encoded) ground truth label to identify region for large deformation
        if self.network_name == 'Unet':
            if self.run_augmentations_in_2D == True:
                roi_region_label = (cropped_label[...,patch_bounds[-1],0] > 0).astype(int)
            else:
                roi_region_label = (cropped_label[...,0] > 0).astype(int)
        else:
            roi_region_label = np.ones((cropped_image_shape)).astype(int)
            if self.run_augmentations_in_2D == True:
                roi_region_label = (roi_region_label[...,patch_bounds[-1]] > 0).astype(int)
        if self.crop_input_image_for_augmentation[0] == True:
            slice_indexing = [slice(pad_amounts[0] if pad_amounts[0]!=0 else None, -pad_amounts[1] if pad_amounts[1]!=0 else None) for pad_amounts in padding_amounts_for_meshgrid[:-1]]
            roi_region_label = roi_region_label[tuple(slice_indexing)]
        #create random deformation field from mean zero standard deviation one gaussian
        if self.run_augmentations_in_2D == True:
            num_dims = self.ndims - 1
        else:
            num_dims = self.ndims
        roi_region_label_shape = roi_region_label.shape
        deformation_field = np.random.normal(size = (num_dims,) + roi_region_label_shape).astype(np.float32)
        #shrink/enlarge deformations in normal/abnormal regions of the brain (use only first channel of ground truth patch)
        deformation_field = (~roi_region_label * self.deform_mesh[1][0] * deformation_field) + (roi_region_label * self.deform_mesh[1][1] * deformation_field)
        #smooth the field with a gaussian
        for i in range(0, num_dims):
            deformation_field[i,...] = gaussian_filter(deformation_field[i,...], self.deform_mesh[2], truncate=self.deform_mesh[3])
        #add in empty channel if only augmenting 2D slices
        if self.run_augmentations_in_2D == True:
            empty_channel = np.zeros(((1,) + roi_region_label_shape))
            deformation_field = np.concatenate([deformation_field, empty_channel], axis=0)
        #zero pad deformation field so that can add back to mesh grid
        deformation_field = np.pad(deformation_field, [pad_bounds[-1], *pad_bounds[:-1]])
        #compose deformation field with inputted meshgrid
        return meshgrid_array + np.array([np.reshape(deformation_field[i], (-1,)) for i in range(0, self.ndims)]).transpose(1,0)
    
    #function to perform affine transforms (scale, rotate, shear)
    def compose_transforms_mesh_grid(self, meshgrid_array, do_augmentation):
        if self.ndims == 2 or self.run_augmentations_in_2D == True:
            affine_transform_size = 2
        else:
            affine_transform_size = self.ndims
        scale_matrix, rotation_matrix, shear_matrix = np.eye(affine_transform_size), np.eye(affine_transform_size), np.eye(affine_transform_size)
        #generate scale transform if requested
        if do_augmentation[1] == True:
            scale_factors = np.random.uniform(*self.scale_mesh[1], affine_transform_size)
            if self.scale_mesh[2] == True:
                scale_factors[1:] = scale_factors[0]
            scale_matrix = scale_matrix * scale_factors
        #generate rotation transform if requested
        if do_augmentation[2] == True:
            rotate_factors = np.random.uniform(*self.rotate_mesh[1], affine_transform_size)
            if affine_transform_size == 2:
                trig_values = [np.cos(rotate_factors[0]), np.sin(rotate_factors[0])]
                rotation_matrix = np.array([[trig_values[0], -trig_values[1]], [trig_values[1], trig_values[0]]])
            else:
                rotation_scipy_obj = Rotation.from_euler('zyx', angles=rotate_factors)
                rotation_matrix = rotation_scipy_obj.as_matrix()
        #generate shear transform if requested
        if do_augmentation[3] == True:
            if affine_transform_size == 2:
                shear_factors = np.random.uniform(*self.shear_mesh[1], affine_transform_size)
            else:
                shear_factors = np.random.uniform(*self.shear_mesh[1], 2 * affine_transform_size)
            if self.shear_mesh[2] == False:
                shear_matrix[np.where(shear_matrix==0)] = shear_factors
            else:
                shear_matrix[tuple(random.sample(list(np.arange(0,affine_transform_size)), 2))] = shear_factors[0]
        #compose transforms together
        if self.randomize_order_augmentations == True:
            random_transform_order = random.sample(list(np.arange(0,3)), 3)
            temp_transforms = np.stack([scale_matrix, rotation_matrix, shear_matrix])
            affine_transform_matrix = np.matmul(np.matmul(temp_transforms[random_transform_order[0],...], temp_transforms[random_transform_order[1],...]), temp_transforms[random_transform_order[2],...])
        else:
            affine_transform_matrix = np.matmul(np.matmul(scale_matrix, rotation_matrix), shear_matrix)
        #apply new affine matrix to meshgrid
        if self.ndims == 3 and self.run_augmentations_in_2D == True:
            meshgrid_array[:, 0:2] = np.dot(meshgrid_array[:, 0:2], affine_transform_matrix)
        else:
            meshgrid_array = np.dot(meshgrid_array, affine_transform_matrix)
        return meshgrid_array
    
    #function to apply gamma correction to cropped volumes
    def gamma_correct_volume(self, cropped_image, cropped_nonzero_image_channel):
        for i in range(0, cropped_image.shape[-1]):
            #randomly choose if to apply gamma correction to this channel
            if np.random.random() < self.augment_data[1][1]:
                cropped_image_channel_view = cropped_image[...,i]
                #only apply correction if channel is not empty (in case are using channel dropout during training) or not a label map
                if cropped_nonzero_image_channel[i] == True and self.input_segmentation_masks[i] == False:
                    #choose whether to brighten (i.e. gamma less than 1) or darken (i.e. gamma greater than 1)
                    if np.random.randint(2) == 1:
                        gamma_factors = np.random.uniform(self.gamma_correction[1][0], 1)
                    else:
                        gamma_factors = np.random.uniform(1, self.gamma_correction[1][1])
                    #only need to transform nonzero values (i.e. ignore outside skull-stripped regions)
                    index_nonzero = np.nonzero(cropped_image_channel_view)
                    #rescale input channel intensity range between 0 and 1 and apply gamma correction
                    gamma_corrected_patch = np.copy(cropped_image_channel_view)
                    min_intensity = np.min(cropped_image_channel_view[index_nonzero])
                    max_intensity = np.max(cropped_image_channel_view[index_nonzero])
                    gamma_corrected_patch[index_nonzero] = np.power(((gamma_corrected_patch[index_nonzero] - min_intensity) / (max_intensity - min_intensity)), gamma_factors)
                    #rescale back to the original intensity distribution
                    gamma_corrected_patch[index_nonzero] = ((max_intensity - min_intensity) * gamma_corrected_patch[index_nonzero]) + min_intensity
                    cropped_image[...,i] = gamma_corrected_patch
        return cropped_image
    
    #function to apply random erosions/dilations and drop connected components from label maps if they are inputted to network
    def erode_dilate_connected_components_input_label_map(self, cropped_image, cropped_nonzero_image_channel):
        for i in range(0, cropped_image.shape[-1]):
            #only apply transforms if channel is not empty (in case are using channel dropout during training) and channel is a label map
            if self.input_segmentation_masks[i] == True and cropped_nonzero_image_channel[i] == True and self.augment_data[0] == True:
                cropped_image_channel_view = cropped_image[...,i]
                transform_functions = [self.erode_input_label_map, self.dilate_input_label_map, self.connected_components_input_label_map]
                random.shuffle(transform_functions)
                for transform_function in transform_functions:
                    cropped_image_channel_view = transform_function(cropped_image_channel_view)
                cropped_image[...,i] = cropped_image_channel_view
        return cropped_image
    
    #helper function to erode label map
    def erode_input_label_map(self, cropped_image_channel_view):
        if self.erode_dilate_conn_comp[0][0] == True and np.random.random() < self.augment_data[1][-3]:
            return binary_erosion(cropped_image_channel_view, structure=self.structuring_element, iterations=np.random.randint(1,self.erode_dilate_conn_comp[0][2]+1))
        else:
            return cropped_image_channel_view
    
    #helper function to dilate label map
    def dilate_input_label_map(self, cropped_image_channel_view):
        if self.erode_dilate_conn_comp[1][0] == True and np.random.random() < self.augment_data[1][-2]:
            return binary_dilation(cropped_image_channel_view, structure=self.structuring_element, iterations=np.random.randint(1,self.erode_dilate_conn_comp[1][2]+1))
        else:
            return cropped_image_channel_view
    
    #helper function to erode label map
    def connected_components_input_label_map(self, cropped_image_channel_view):
        if self.erode_dilate_conn_comp[2][0] == True:
            conn_comp = scipy_conn_comp(cropped_image_channel_view, structure=self.structuring_element)
            prob_delete_component = 0
            if conn_comp[1] >= 1:
                #normalize deletion probability so that label maps with different number of connected components still have similar overall deletion rates
                prob_delete_component = max(0.01, 1 - ((1 - self.augment_data[1][-1]) ** (1 / conn_comp[1])))
            for lesion_num in range(1, conn_comp[1]+1):
                if np.random.random() < prob_delete_component:
                    conn_comp[0][conn_comp[0] == lesion_num] = 0
            return (conn_comp[0] > 0).astype(int)
        else:
            return cropped_image_channel_view

    #function to apply mirroring (with 50% probability) over requested axes
    def mirror_patch(self, patched_image, patched_label):
        for i in range(0, len(self.left_right_flip[1])):
            if np.random.random() < self.augment_data[1][0]:
                patched_image = np.flip(patched_image, axis=self.left_right_flip[1][i])
                if self.network_name == 'Unet':
                    patched_label = np.flip(patched_label, axis=self.left_right_flip[1][i])
        return patched_image, patched_label

    #helper function to determine patch and pad bounds
    def get_patch_and_pad_bounds(self, center_index, volume_shape, correct_patch_regions=False, true_patch_regions=None):
        patch_bounds, pad_bounds = [], []
        if self.full_image_patching == True:
            if true_patch_regions == None:
                patient_patch_regions = [int(x / 2) for x in volume_shape]
            else:
                patient_patch_regions = [int(x / 2) for x in true_patch_regions]
        else:
            patient_patch_regions = deepcopy(self.patch_regions)
        patch_regions_corrected = patient_patch_regions
        if correct_patch_regions == True and self.voxel_spacing[0] == False:
            patch_regions_corrected = (patient_patch_regions * self.voxel_spacing[1]).astype(int)
        for ind, reg, dim_shape in zip(center_index, patch_regions_corrected, volume_shape):
            min_index = ind - reg
            max_index = ind + reg
            pad_min, pad_max = 0, 0
            if min_index < 0:
                pad_min = -min_index
                min_index = 0
            if max_index > dim_shape:
                pad_max = max_index - dim_shape
                max_index = dim_shape
            patch_bounds.append(slice(min_index, max_index, 1))
            pad_bounds.append(tuple([pad_min, pad_max]))
        #No padding for channel dimension
        pad_bounds.append(tuple([0, 0]))
        return tuple(patch_bounds), tuple(pad_bounds)
    
    #helper function to randomly patch within the confines of the cropped volume shape
    def get_center_index_from_volume_shape(self, volume_shape):
        center_index = []
        if self.full_image_patching == True:
            patient_patch_regions = [int(x / 2) for x in volume_shape]
        else:
            patient_patch_regions = deepcopy(self.patch_regions)
        for reg, dim_shape in zip(patient_patch_regions, volume_shape):
            random_translation = 0
            dim_shape = dim_shape / 2
            extra_space = dim_shape - reg
            if extra_space >= 0:
                random_translation = np.random.randint(extra_space + 1)
            center_index.append(int(dim_shape + random_translation))
        return center_index

    #helper function to load and crop patients
    def load_and_crop(self, patient):
        images = self.load_volumes(patient, self.input_image_names)
        #extract bounding box region of brain (sum over all input channels to find global FOV) to increase speed of downstream computation
        volume_location = find_objects(np.sum(images,axis=-1) != 0)[0]
        cropped_image = images[volume_location]
        #figure out which images/labels are empty (in case using channel drop or have normal patients with no tumor segmentations)
        cropped_nonzero_image_channel = [False] * cropped_image.shape[-1]
        for i in range(0, cropped_image.shape[-1]):
            if np.any(cropped_image[...,i]):
                cropped_nonzero_image_channel[i] = True
        #extract labels differently depending on if this is for segmentation or classification
        if self.network_name == 'Unet':
            labels = self.load_volumes(patient, self.ground_truth_label_names)
            cropped_label = labels[volume_location]
            cropped_nonzero_label_channel = [False] * cropped_label.shape[-1]
            for i in range(0, cropped_label.shape[-1]):
                if np.any(cropped_label[...,i]):
                    cropped_nonzero_label_channel[i] = True
        else:
            cropped_label = self.load_classification(patient, self.ground_truth_label_names)
            cropped_nonzero_label_channel = None
        return cropped_image, cropped_label, cropped_nonzero_image_channel, cropped_nonzero_label_channel

    #helper function to cache patients into a dictionary
    def cache_patients_into_memory(self):
        for patient in self.patients:
            #load patient data and crop volumes
            cropped_image, cropped_label, cropped_nonzero_image_channel, cropped_nonzero_label_channel = self.load_and_crop(patient)
            self.data_cache[patient] = (cropped_image, cropped_label, cropped_nonzero_image_channel, cropped_nonzero_label_channel)
            if self.run_augmentations_every_epoch[0] == False:
                self.augmented_data_cache[patient] = []
        with open(self.output_file, 'a') as f:
            f.write('Cached ' + self.name + ' dataset \n')

    #helper function to load volumes from list
    def load_volumes(self, patient, volume_name_list):
        patient_dir = self.data_dir + patient + '/'
        flat_list_len = len(self.flatten_list(volume_name_list))
        volume_preallocated = False
        all_volumes = None
        counter = 0
        for name in volume_name_list:
            if not isinstance(name, list):
                if name != None and os.path.exists(patient_dir + name):
                    #round data to 5 decimal points to ensure that cropping to non-zero area works properly!
                    image = np.round(nib.load(patient_dir + name).get_data().astype(np.float32), 5)
                else:
                    image = None
            else:
                image = self.load_volumes(patient, name)
                if image is None:
                    counter = counter + len(name)
            if volume_preallocated == False and image is not None:
                all_volumes = np.zeros((*image.shape[:self.ndims], flat_list_len))
                volume_preallocated = True
            if image is not None:
                if len(image.shape) == self.ndims:
                    all_volumes[...,counter] = image
                    counter = counter + 1
                else:
                    all_volumes[...,counter:counter+image.shape[-1]] = image
                    counter = counter + image.shape[-1]
            elif not isinstance(name, list):
                counter = counter + 1
        return all_volumes
    
    #helper function to load classification outputs from list
    def load_classification(self, patient, volume_name_list):
        patient_dir = self.data_dir + patient + '/'
        all_class_outputs = np.zeros((len(volume_name_list), 1))
        for i, name in enumerate(volume_name_list):
            with open(patient_dir + name, 'r') as f:
                class_output = int(f.read())
            all_class_outputs[i,0] = class_output
        return all_class_outputs
    
    #helper function to balance classification training dataset at the end of each epoch
    def balance_classification_dataset(self):
        epoch_classification_labels_array = np.array(list(np.repeat(self.classification_labels_array, self.num_patches_per_patient, axis=0)) + random.sample(list(self.classification_labels_array), self.pad_samples))
        positive_cases = list(epoch_classification_labels_array[epoch_classification_labels_array[:,0] == 1, 1])
        negative_cases = list(epoch_classification_labels_array[epoch_classification_labels_array[:,0] == 0, 1])
        if self.balanced_epochs_for_classification[0] == True:
            #sample with replacement from the positive class
            positive_cases = positive_cases + random.choices(positive_cases, k=self.balanced_epochs_for_classification[1])
            #sample without replacement from the negative class
            negative_cases = random.sample(negative_cases, k=(len(negative_cases) - self.balanced_epochs_for_classification[1]))
        if self.balanced_batches_for_classification == True:
            #number of positive cases per batch
            num_positive_cases_per_batch_array = np.zeros(self.num_batches_per_epoch)
            num_pos_per_batch, num_pos_extra = len(positive_cases) // self.num_batches_per_epoch, len(positive_cases) % self.num_batches_per_epoch
            num_positive_cases_per_batch_array[:num_pos_extra] = num_pos_per_batch + 1
            num_positive_cases_per_batch_array[num_pos_extra:] = num_pos_per_batch
            random.shuffle(num_positive_cases_per_batch_array)
            #number of negative cases per batch
            num_negative_cases_per_batch_array = self.batch_size - num_positive_cases_per_batch_array
            #randomly sample without replacement from list of positive/negative cases to make final patients list
            patient_indexes = []
            for num_positive_cases_per_batch, num_negative_cases_per_batch in zip(num_positive_cases_per_batch_array, num_negative_cases_per_batch_array):
                temp_batch_positive = []
                for _ in range(0, int(num_positive_cases_per_batch)):
                    temp_batch_positive.append(positive_cases.pop(random.randint(0,len(positive_cases)-1)))
                temp_batch_negative = []
                for _ in range(0, int(num_negative_cases_per_batch)):
                    temp_batch_negative.append(negative_cases.pop(random.randint(0,len(negative_cases)-1)))
                temp_batch = temp_batch_positive + temp_batch_negative
                random.shuffle(temp_batch)
                patient_indexes.extend(temp_batch)
            self.patients_list_all = [self.patients[int(patient_index)] for patient_index in patient_indexes]
        else:
            patient_indexes = positive_cases + negative_cases
            self.patients_list_all = [self.patients[int(patient_index)] for patient_index in patient_indexes]
            random.shuffle(self.patients_list_all)
    
    #helper function to flatten nested list
    def flatten_list(self, nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(self.flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list
    
    #helper function to choose proper modalities
    def randomly_choose_inputs(self, cropped_volume, cropped_nonzero_volume_channel, volume_name_list):
        cropped_volume_final = np.zeros((*cropped_volume.shape[:-1], len(volume_name_list)))
        cropped_nonzero_volume_channel_final = [False] * len(volume_name_list)
        counter = 0
        for i, name in enumerate(volume_name_list):
            if not isinstance(name, list):
                cropped_volume_final[...,i] = cropped_volume[...,counter]
                cropped_nonzero_volume_channel_final[i] = cropped_nonzero_volume_channel[counter]
                counter = counter + 1
            else:
                num_choices = len(name)
                #if None not in volume_name_list, only choose non-zero channel to ensure are always training with real data
                prob_choose_channel = [True] * num_choices
                if None not in name:
                    prob_choose_channel = [True if cropped_nonzero_volume_channel[counter+ind] else False for ind in range(0,num_choices)]
                prob_choose_channel = prob_choose_channel / np.sum(prob_choose_channel)
                random_index = np.random.choice(num_choices, p = prob_choose_channel)
                cropped_volume_final[...,i] = cropped_volume[...,counter+random_index]
                cropped_nonzero_volume_channel_final[i] = cropped_nonzero_volume_channel[counter+random_index]
                counter = counter + num_choices
        return cropped_volume_final, cropped_nonzero_volume_channel_final
            
#helper function to get all filepaths if there are nested folders (and only choose folders that have all the necessary volumes)
def nested_folder_filepaths(patient_dir, vols_to_process=None):
    if vols_to_process == None:
        relative_filepaths = [os.path.relpath(directory_paths, patient_dir) for (directory_paths, directory_names, filenames) in os.walk(patient_dir) if len(filenames)!=0]
    else:
        vols_to_process = [item for sublist in vols_to_process for item in sublist]
        relative_filepaths = [os.path.relpath(directory_paths, patient_dir) for (directory_paths, directory_names, filenames) in os.walk(patient_dir) if check_necessary_files_exist(filenames, vols_to_process)]
    relative_filepaths.sort()
    return relative_filepaths

#helper function to check if patient folder has all necessary files
def check_necessary_files_exist(filenames, vols_to_process):
    all_files_exist = np.zeros(len(vols_to_process))
    filenames.append(None)
    for i, vol_to_process in enumerate(vols_to_process):
        #this volume is necessary and must be present in the patient folder
        if not isinstance(vol_to_process, list):
            if vol_to_process in filenames:
                all_files_exist[i] = True
            else:
                all_files_exist[i] = False
        else:
            if any(exchangeable_vols in filenames for exchangeable_vols in vol_to_process):
                all_files_exist[i] = True
            else:
                all_files_exist[i] = False
    if np.all(all_files_exist):
        return True
    else:
        return False