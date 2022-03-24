import os
import errno
import shutil
import numpy as np
import nibabel as nib

from subprocess import call
from nipype.interfaces.ants import N4BiasFieldCorrection

#Function to convert all dicoms to Nifti format
def convert_dicom_to_nifti(dicom_dir, nifti_dir, patient, dcm2niix_dir, output_vol_name=None):
    #filepath to input dicom folder
    dicom_folder = dicom_dir + patient
    os.chdir(dicom_folder)
    #make output patient folder if it does not exist
    output_nifti_folder = nifti_dir + patient
    if not os.path.exists(output_nifti_folder):
        os.makedirs(output_nifti_folder)
    if output_vol_name == None:
        #grab all folders in dicom directory
        dicom_volumes = next(os.walk('.'))[1]
        for dicom_volume in dicom_volumes:
            input_dicom_volume_folder = dicom_folder + '/' + dicom_volume
            convert_command = [dcm2niix_dir + ' -z y -f ' + dicom_volume + ' -o "' + output_nifti_folder + '" "' + input_dicom_volume_folder + '"']
            call(' '.join(convert_command), shell=True)
    else:
        convert_command = [dcm2niix_dir + ' -z y -f ' + output_vol_name + ' -o "' + output_nifti_folder + '" "' + dicom_folder + '"']
        call(' '.join(convert_command), shell=True)
    #return created file names
    os.chdir(output_nifti_folder)
    output_filenames = next(os.walk('.'))[2]
    return output_filenames

#Function to change all images to desired orientation
def reorient_volume(nifti_dir, patient, vols_to_process, orientation, slicer_dir):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, orientation)
    #orientation module
    module_name = 'OrientScalarVolume'
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        orientation_command = [slicer_dir, '--launch', module_name,  '"' + input_filepath + '" "' + output_filepath + '"', '-o', orientation]
        call(' '.join(orientation_command), shell=True)
    #return created file names
    return output_filenames

#Function to compute affine registration between moving (low res scan) and fixed (high res scan that you are registering all other sequences to) volume
def register_volume(nifti_dir, patient, fixed_volume, vols_to_process, transform_mode, transform_type, interpolation_mode, sampling_percentage, output_transform_filename, slicer_dir, use_external_reference=False, append_tag='REG'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    if use_external_reference == False:
        input_fixed_vol_filepath, output_fixed_vol_filepath, input_others_filepaths, output_others_filepaths = choose_volume(fixed_volume, vols_to_process, input_filepaths, output_filepaths)
    else:
        input_fixed_vol_filepath, output_fixed_vol_filenames, output_fixed_vol_filepath = generate_filepaths(nifti_dir, patient, [fixed_volume], '')
        input_others_filepaths = input_filepaths
        output_others_filepaths = output_filepaths
    #rename the fixed volume for consistency
    os.rename(input_fixed_vol_filepath[0], output_fixed_vol_filepath[0])
    for i, (input_others, output_others) in enumerate(zip(input_others_filepaths, output_others_filepaths)):
        if len(input_others_filepaths) > 1:
            temp_output_transform_filename = output_transform_filename[:-3] + '_' + str(i) + '.h5'
        else:
            temp_output_transform_filename = output_transform_filename
        affine_registration_command = [slicer_dir,'--launch', 'BRAINSFit', '--fixedVolume', '"' + output_fixed_vol_filepath[0] + '"', '--movingVolume', '"' + input_others + '"', '--transformType', transform_type, '--initializeTransformMode', transform_mode, '--interpolationMode', interpolation_mode, '--samplingPercentage', str(sampling_percentage), '--outputTransform', temp_output_transform_filename, '--outputVolume', output_others]
        call(' '.join(affine_registration_command), shell=True)
    #return created file names
    return output_filenames

#Function to resample all volumes to desired spacing
def resample_volume(nifti_dir, patient, vols_to_process, spacing, interp_type, slicer_dir, append_tag='RESAMPLED'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    #resampling module
    module_name = 'ResampleScalarVolume'
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        resample_scalar_volume_command = [slicer_dir,'--launch', module_name, '"' + input_filepath + '" "' + output_filepath + '"', '-i', interp_type, '-s', spacing]
        call(' '.join(resample_scalar_volume_command), shell=True)
    #return created file names
    return output_filenames

#Function to resample all volumes using a reference volume (or transform file)
def resample_volume_using_reference(nifti_dir, patient, vols_to_process, reference_volume, interp_type, slicer_dir, output_transform_filename=None, centered_transform=None, manual_transform='1,0,0,0,1,0,0,0,1,0,0,0', append_tag='RESAMPLED'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    #resampling module
    module_name = 'ResampleScalarVectorDWIVolume'
    if reference_volume != None:
        reference_volume_filepath = nifti_dir + patient + '/' + reference_volume
    if interp_type == 'nearestNeighbor' or interp_type == 'nn':
        interp_type = 'nn'
    else:
        interp_type = 'bs'
    for i, (input_filepath, output_filepath) in enumerate(zip(input_filepaths, output_filepaths)):
        if reference_volume == None and output_transform_filename == None:
            resample_scalar_volume_command = [slicer_dir,'--launch', module_name, '"' + input_filepath + '" "' + output_filepath + '"', '-i', interp_type, '-m', manual_transform]
        elif reference_volume != None and output_transform_filename == None:
            resample_scalar_volume_command = [slicer_dir,'--launch', module_name, '"' + input_filepath + '" "' + output_filepath + '"', '-i', interp_type, '-R', reference_volume_filepath]
        else:
            if len(input_filepaths) > 1 and not os.path.exists(nifti_dir + patient + '/' + output_transform_filename):
                temp_output_transform_filename = output_transform_filename[:-3] + '_' + str(i) + '.h5'
            else:
                temp_output_transform_filename = output_transform_filename
            if reference_volume == None:
                resample_scalar_volume_command = [slicer_dir,'--launch', module_name, '"' + input_filepath + '" "' + output_filepath + '"', '-i', interp_type, '-f', temp_output_transform_filename]
            else:
                resample_scalar_volume_command = [slicer_dir,'--launch', module_name, '"' + input_filepath + '" "' + output_filepath + '"', '-i', interp_type, '-f', temp_output_transform_filename, '-R', reference_volume_filepath]
        if centered_transform != None:
            resample_scalar_volume_command.append('-c')
        call(' '.join(resample_scalar_volume_command), shell=True)
    #return created file names
    return output_filenames

#Function to perform N4 bias correction
def n4_bias_correction(nifti_dir, patient, vols_to_process, n4_iterations, mask_image=None, append_tag='N4'):
	#input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        n4 = N4BiasFieldCorrection(output_image = output_filepath)
        n4.inputs.input_image = input_filepath
        n4.inputs.n_iterations = n4_iterations
        if mask_image != None:
            n4.inputs.mask_image = os.path.join(nifti_dir + patient, mask_image)
        n4.run()
    #return created file names
    return output_filenames

#Function to skull strip volume of choice using ROBEX (and apply skull-strip mask to other volumes)
def skull_strip(nifti_dir, patient, volume_to_skullstrip, vols_to_process, robex_dir, append_tag='SS'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    input_vol_skullstrip_filepath, output_vol_skullstrip_filepath, input_others_filepaths, output_others_filepaths = choose_volume(volume_to_skullstrip, vols_to_process, input_filepaths, output_filepaths)
    #skull stripping operation
    call([robex_dir + ' "' + input_vol_skullstrip_filepath[0] + '" "' + output_vol_skullstrip_filepath[0] + '"'], shell=True)
    #apply skull stripping region to other volumes
    vol_ss = nib.load(output_vol_skullstrip_filepath[0]).get_data()
    zero_vals = vol_ss == 0
    for input_others, output_others in zip(input_others_filepaths, output_others_filepaths):
        #load nifti volume
        nib_vol = nib.load(input_others)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()
        vol[zero_vals] = 0
        nib_vol_ss = nib.Nifti1Image(vol, affine, header=header)
        nib.save(nib_vol_ss, output_others)
    #return created file names
    return output_filenames

def get_non_zero_mask(nifti_dir, patient, vols_to_process, append_tag='mask'):
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
        #load nifti volume
        nib_vol = nib.load(input_filepath)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()
        vol_mask = (vol != 0).astype(np.int)
        nib_vol_mask = nib.Nifti1Image(vol_mask, affine, header=header)
        nib.save(nib_vol_mask, output_filepath)
    #return created file names
    return output_filenames

def replace_affine_header(nifti_dir, patient, vols_to_process, reference_volume, append_tag=''):
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    reference_vol_nib = nib.load(os.path.join(nifti_dir + patient, reference_volume))
    affine = reference_vol_nib.get_affine()
    header = reference_vol_nib.get_header()
    for i, (input_filepath, output_filepath) in enumerate(zip(input_filepaths, output_filepaths)):
        #load nifti volume
        vol = nib.load(input_filepath).get_data()
        vol_new = nib.Nifti1Image(vol, affine, header=header)
        nib.save(vol_new, output_filepath)
    #return created file names
    return output_filenames

def mask_volume(nifti_dir, patient, vols_to_process, mask, append_tag='masked'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    if not isinstance(mask, list):
        mask = [mask] * len(vols_to_process)
    for i, (input_filepath, output_filepath) in enumerate(zip(input_filepaths, output_filepaths)):
        #load in mask
        mask_vol = nib.load(os.path.join(nifti_dir + patient, mask[i])).get_data()
	    #load nifti volume
        nib_vol = nib.load(input_filepath)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()
        vol_masked = vol * mask_vol
        nib_vol_masked = nib.Nifti1Image(vol_masked, affine, header=header)
        nib.save(nib_vol_masked, output_filepath)
    #return created file names
    return output_filenames

#Function to perform normalization (if no mean/std is given, will perform per-volume mean zero, standard deviation one normalization by default); reference volume will be used to generate appropriate skull mask; skull_mask_volume is a shortcut used in ALD pre-processing
def normalize_volume(nifti_dir, patient, vols_to_process, only_nonzero=True, normalization_params=np.array([]), reference_volume=None, skull_mask_volume=None, append_tag='NORM'):
    if len(normalization_params) > 0 and len(normalization_params.shape) == 1:
        normalization_params = np.tile(normalization_params, (len(vols_to_process), 1))
    if reference_volume != None:
        reference_vol = nib.load(os.path.join(nifti_dir + patient, reference_volume)).get_data()
        skull_mask = (reference_vol != 0).astype(np.int)
    if skull_mask_volume != None:
        skull_mask_vol = nib.load(os.path.join(nifti_dir + patient, skull_mask_volume)).get_data()
        skull_mask = (skull_mask_vol != 0).astype(np.int)
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for i, (input_filepath, output_filepath) in enumerate(zip(input_filepaths, output_filepaths)):
	    #load nifti volume
        nib_vol = nib.load(input_filepath)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()
	    #Normalize only non-zero intensity values (if flag set to true)
        if only_nonzero == True and reference_volume == None and skull_mask_volume == None:
            idx_nz = np.nonzero(vol)
        elif only_nonzero == True and (reference_volume != None or skull_mask_volume != None):
            idx_nz = np.nonzero(skull_mask)
        else:
            idx_nz = np.where(vol)
        if len(normalization_params) == 0:
            mean, std = np.mean(vol[idx_nz]), np.std(vol[idx_nz])
        else:
            mean, std = normalization_params[i, :]
        vol_norm = np.copy(vol)
        if reference_volume == None:
            vol_norm[idx_nz] = (vol_norm[idx_nz] - mean) / std
        else:
            vol_norm = (vol_norm - mean) / std
        nib_vol_norm = nib.Nifti1Image(vol_norm, affine, header=header)
        nib.save(nib_vol_norm, output_filepath)
    #return created file names
    return output_filenames

#function to binarize ROI
def binarize_segmentation(nifti_dir, patient, roi_to_process, append_tag='BINARY-label'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, roi_to_process, append_tag)
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
	    #load nifti volume
        nib_roi = nib.load(input_filepath)
        affine = nib_roi.get_affine()
        header = nib_roi.get_header()
        roi = nib_roi.get_data()
	    #binarize non-zero intensity values
        roi[np.nonzero(roi)] = 1
        nib_roi_binary = nib.Nifti1Image(roi, affine, header=header)
        nib.save(nib_roi_binary, output_filepath)
    #return created file names
    return output_filenames

#function to rescale intensity range (if no range is given, will default to rescaling range to [0,1])
def intensity_rescale_volume(nifti_dir, patient, vols_to_process, rescale_range=np.array([0,1]), append_tag='RESCALED'):
    if len(rescale_range.shape) == 1:
        rescale_range = np.tile(rescale_range, (len(vols_to_process), 1))
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for i, (input_filepath, output_filepath) in enumerate(zip(input_filepaths, output_filepaths)):
	    #load nifti volume
        nib_vol = nib.load(input_filepath)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()
	    #rescale intensities to new min and max
        old_min, old_max = np.min(vol), np.max(vol)
        new_min, new_max = rescale_range[i, :]
        rescaled_vol = (vol - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
        nib_rescaled_vol = nib.Nifti1Image(rescaled_vol, affine, header=header)
        nib.save(nib_rescaled_vol, output_filepath)
    #return created file names
    return output_filenames

#function to change extremely small values to 0 (in case there were rounding errors during resampling/resizing operations)
def round_volume(nifti_dir, patient, vols_to_process, decimals=5, append_tag=''):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
	    #load nifti volume
        nib_vol = nib.load(input_filepath)
        affine = nib_vol.get_affine()
        header = nib_vol.get_header()
        vol = nib_vol.get_data()
	    #round small intensities to zero
        vol[np.abs(vol) < (10**(-decimals))] = 0
        nib_round_vol = nib.Nifti1Image(vol, affine, header=header)
        nib.save(nib_round_vol, output_filepath)
    #return created file names
    return output_filenames

#function to rename files
def rename_volumes(nifti_dir, patient, vols_to_process, save_names, append_tag=''):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag, save_names=save_names)
    for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
	    #rename volumes
        os.rename(input_filepath, output_filepath)
    #return created file names
    return output_filenames

#function to generate ADC map from B0 and B1000 image
def create_adc_volume(nifti_dir, patient, vols_to_process, append_tag='ADC'):
    #input/output filepaths
    input_filepaths, output_filenames, output_filepaths = generate_filepaths(nifti_dir, patient, vols_to_process, append_tag)
    for i, input_filepath in enumerate(input_filepaths):
        #load nifti volume
        nib_vol = nib.load(input_filepath)
        vol = nib_vol.get_data()
        if i == 0:
            affine = nib_vol.get_affine()
            header = nib_vol.get_header()
            b_vols = np.zeros(vol.shape + (2,))
            idx_zero = []
        idx_zero.append(np.where(vol <= 0))
        vol[idx_zero[i]] = 1
        b_vols[...,i] = vol
    ADC = np.log(np.divide(b_vols[...,0], b_vols[...,1])) / -1000
    for indexes in idx_zero:
	    ADC[indexes] = 0
    nib_ADC = nib.Nifti1Image(ADC, affine, header=header)
    save_name = append_tag + '.nii.gz'
    nib.save(nib_ADC, nifti_dir + patient + '/' + save_name)
    return [save_name]

#function to threshold probability masks as requested thresholds (will binarize at 0.5 as default with no tags appended to generated label maps)
def threshold_probability_mask(nifti_dir, patient, vols_to_process, thresholds=[0.5]):
    #input/output filepaths
    input_folder = nifti_dir + patient
    input_filepaths = [input_folder + '/' + i for i in vols_to_process]
    for input_filepath in input_filepaths:
        probability_vol, affine, header = load_nifti_volume(input_filepaths=[input_filepath])
        #binarize predicted label map at requested thresholds
        for threshold in thresholds:
            probability_vol_binarized = (probability_vol[...,0] >= threshold).astype(int)
            #save output
            save_name_vol = 'threshold_' + str(threshold) + '_pred-label.nii.gz'
            save_nifti_volume(input_filepath, [save_name_vol], [probability_vol_binarized], affine=affine, header=header)

#function to get all filepaths if there are nested folders (and only choose folders that have all the necessary volumes)
def nested_folder_filepaths(nifti_dir, vols_to_process=None):
    if vols_to_process == None:
        relative_filepaths = [os.path.relpath(directory_paths, nifti_dir) for (directory_paths, directory_names, filenames) in os.walk(nifti_dir) if len(filenames)!=0]
    else:
        relative_filepaths = [os.path.relpath(directory_paths, nifti_dir) for (directory_paths, directory_names, filenames) in os.walk(nifti_dir) if all(vol_to_process in filenames for vol_to_process in vols_to_process)]
    return relative_filepaths 

#function to copy files (if output names are not given, will use source directory names)
def copy_and_move_files(input_dir, output_dir, file_names, output_file_names=None):
    input_filepaths = [input_dir + '/' + i for i in file_names]
    if all(os.path.exists(input_filepath) for input_filepath in input_filepaths):
        if output_file_names == None:
            output_file_names = file_names
        output_filepaths = [output_dir + '/' + i for i in output_file_names]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for input_filepath, output_filepath in zip(input_filepaths, output_filepaths):
            shutil.copy(input_filepath, output_filepath)

#function to copy entire folder and subdirectories to new location
def copy_and_move_folders(input_dir, output_dir):
    try:
        shutil.copytree(input_dir, output_dir)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(input_dir, input_dir)
        else:
            print('Directory not copied. Error: %s' % e)

#helper function to load nifti volumes as numpy arrays (along with affine and header information if requested)
def load_nifti_volume(input_filepaths=None, vols_to_process=None, load_affine_header=True):
    if vols_to_process != None:
        input_filepaths = [input_filepaths + vol_to_process for vol_to_process in vols_to_process]
    for j, input_filepath in enumerate(input_filepaths):
        nib_vol = nib.load(input_filepath)
        image = nib_vol.get_data()
        if j == 0:
            affine = nib_vol.get_affine()
            header = nib_vol.get_header()
            all_volumes = np.zeros((image.shape) + ((len(input_filepaths),)))
        all_volumes[...,j] = image
    if load_affine_header == True:
        return all_volumes, affine, header
    else:
        return all_volumes

#helper function to save numpy arrays as nifti volumes (using affine and header if given)
def save_nifti_volume(input_filepath, save_names, numpy_volume_list, affine=None, header=None):
    for j, (save_name, save_vol) in enumerate(zip(save_names, numpy_volume_list)):
        if header==None and affine==None:
            affine = np.eye(len(save_vol.shape) + 1)
            save_vol_nib = nib.Nifti1Image(save_vol, affine)
        else:
            save_vol_nib = nib.Nifti1Image(save_vol, affine, header=header)
        nib.save(save_vol_nib, input_filepath + save_name)

#helper function to generate input/output filepaths
def generate_filepaths(data_dir, patient_name, vols_to_process, append_tag, save_names=None):
    #filepath to input patient folder
    input_folder = data_dir + patient_name
    os.chdir(input_folder)
    #filepath to volumes
    input_filepaths = [input_folder + '/' + i for i in vols_to_process]
    #output volume names and file paths
    if save_names != None:
        output_filenames = save_names
        output_filepaths = [input_folder + '/' + i for i in output_filenames]
    elif append_tag == '' or append_tag == None:
        output_filenames = vols_to_process
        output_filepaths = input_filepaths
    else:
        output_filenames = [i[:i.find('.nii')] + '_' + append_tag + '.nii.gz' for i in vols_to_process]
        output_filepaths = [input_folder + '/' + i for i in output_filenames]
    return input_filepaths, output_filenames, output_filepaths

#helper function to find volume of interest from list of volumes to process
def choose_volume(vol_special, vols_to_process, input_filepaths, output_filepaths):
    index_vol_special = np.array([vol_special in vol for vol in vols_to_process]) 
    input_vol_special_path = [i for (i, j) in zip(input_filepaths, index_vol_special) if j]
    output_vol_special_path = [i for (i, j) in zip(output_filepaths, index_vol_special) if j]
    input_vol_paths = [i for (i, j) in zip(input_filepaths, ~index_vol_special) if j]
    output_vol_paths = [i for (i, j) in zip(output_filepaths, ~index_vol_special) if j]
    return input_vol_special_path, output_vol_special_path, input_vol_paths, output_vol_paths