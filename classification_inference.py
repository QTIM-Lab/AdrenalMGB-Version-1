import os
import numpy as np

from train_cluster import *
from predict_cluster import *

#axis 0: left-right flip (i.e. spinal cord stays in place)
#axis 1: front-back flip (i.e. spinal cord moves locations)
#axis 2: top-bottom flip (i.e. lungs move locations)

trainModel = False   #set to false if you want to predict on the train/val/test set using this config file

params_dict = {}
params_dict['random_seed'] = [False, 0]      #whether to use a random seed (for numpy and tensorflow operations), and what the seed number should be
#project directory
params_dict['scratch_dir'] = str(os.environ['SLURM_JOB_SCRATCHDIR']) + '/'
params_dict['job_id'] = str(os.environ["SLURM_JOB_ID"])
#train/val/test directories
params_dict['data_dir_train'] = params_dict['scratch_dir'] + 'Train/'      
params_dict['data_dir_val'] = params_dict['scratch_dir'] + 'Val/'
params_dict['data_dir_test'] = params_dict['scratch_dir'] + 'Test/'
params_dict['data_dirs_predict'] = [params_dict['data_dir_train'], params_dict['data_dir_val'], params_dict['data_dir_test']]
#input/output and model file names
params_dict['input_image_names'] = ['ct_soft_tissue_ref_pred_crop.nii.gz', 'predicted_ref_pred_crop-label.nii.gz']         #list of input volumes names
params_dict['ground_truth_label_names'] = ['adrenal_class.txt']                                          #list of ground truth volumes names
params_dict['input_segmentation_masks'] = [False, True]                                                  #if input list includes segmentation mask (as is done in cascaded networks), pass in list to show which elements are masks (i.e. if input is MRI and mask, then this variable is [False, True])
params_dict['model_outputs_dir'] = params_dict['scratch_dir'] + 'model_outputs_classification/'                         #directory where all outputs of the model will be saved (i.e. saved model weights, optimizer, text file, etc.)
params_dict['model_weights_save_path'] = params_dict['model_outputs_dir'] + 'tf_ckpts'                   #model weights name used when saving/loading a model to/from memory
params_dict['max_number_checkpoints_keep'] = 5                                                           #number of most recent models to save in tensorflow checkpoint format (models saved based off of validation monitor value)
params_dict['model_config_save_path'] = params_dict['model_outputs_dir'] + 'config.pkl'                  #model config so that can perfectly re-create network
params_dict['predicted_label_name'] = ['model_prediction.txt', 'model_probability.txt']                  #binarized predicted output save name and probability mask save name
params_dict['output_file'] = params_dict['model_outputs_dir'] + params_dict['job_id'] + '.txt'           #name of text file containing epoch loss/metrics
params_dict['tensorboard_dir'] = params_dict['model_outputs_dir'] + params_dict['job_id']                #name of logs directory for tensorboard
params_dict['model_image_save_path'] = params_dict['model_outputs_dir'] + 'model_image'                  #name of model image that will be saved out
#train params
params_dict['reload_model'] = [False, False]        #whether to reload a previous model, and whether to use previous model state (i.e. optimizer params) when resuming training
params_dict['patch_size'] = [64,64,16]              #size of patches to get from input volumes
params_dict['sample_background_prob'] = 0.5         #probability of sampling background class during patching
params_dict['batch_size'] = [16, 16, 16]            #batch size during training, validation, and prediction
params_dict['learning_rate'] = 0.1                  #initial learning rate 
params_dict['weight_decay'] = 0.0005                #initial weight decay (only used for optimizers that accept weight decay parameter -> AdamW and SGDW)
params_dict['num_patches_per_patient'] = [1, 16]                        #number of patches to extract from each patient in the training set and validation set
params_dict['adaptive_full_image_patching'] = [False, False]            #whether to train on full size images, and whether to validate on full size images (in order to get accurate validation set dice score)
params_dict['use_conn_comp_patching'] = False                           #whether to patch from set of connected components (only set to True if have binary task with many distinct lesions)
params_dict['patch_inside_volume'] = True                               #whether to patch inside the volume region for normal patches (which should help stabalize training by using patches with limited added padding in it)
params_dict['patch_center_of_tumor_for_classification'] = False         #whether to create a patch using the center of the tumor segmentation (only usable if tumor segmentation given as input to the network)
params_dict['require_positive_class_per_batch'] = False                 #whether all batches are required to have an example of the positive class present (will help stabilize training especially when training at small batch sizes)
params_dict['balanced_batches_for_classification'] = False              #whether to balance classes such that each batch has approximiately the same percentage of positive to negative examples
params_dict['balanced_epochs_for_classification'] = [True, 40]          #whether to balance classes such that each epoch has the same number of total examples, but has n number of negatives randomly swapped out for duplicated positives
params_dict['run_augmentations_every_epoch'] = [True, 1]                #whether to run augmentations every epoch; if False, how many epochs before refreshing data cache; only possible if "cache_volumes" option is true
params_dict['voxel_spacing'] = [False, np.array([1,1,5])]               #whether your data has isotropic voxel spacing (if it does, you can ignore the second option; otherwise put voxel spacing to ensure augmentations are performed correctly)
params_dict['downsample_for_augmentation'] = [False, 2]                 #whether to downsample the meshgrid for faster augmentation (helpful if augmentation is bottle-necking training script), and if so, the factor by which to downsample the grid
params_dict['run_augmentations_in_2D'] = False                          #whether to augment 2D slices in 3D volume instead of using native 3D augmentations (if using highly anisotropic data, augmentations can take lots of time to run since need to resample data (and can be subotimal due to the anisotropic nature of the data))
params_dict['crop_input_image_for_augmentation'] = [False, 5, False]    #whether to crop the input images when augmenting the images (will speed up augmentation, but may cause artifacts in patches); amount of voxels to keep around patch to limit artifacts; whether to account for voxel spacing and use the downsample_for_augmentation parameter (suggested to be False)
params_dict['probability_augmentation'] = [.5,.5,.1,.5,.5,.5,.1]                         #probability of applying each of the requested transforms (if supplying list of probabilities, must be of length 7 (one probability for flip, gamma, deform, scale, rotate, shear, translate))
params_dict['augmentation_scheduler'] = [True, 25]                                       #whether to ramp up augmentation probability over many epochs; number of epochs to reach full augmentation probability
params_dict['left_right_flip'] = [False, (0,1,2)]                                        #whether to apply left/right patch flipping augmentation, and axes along which to perform flip
params_dict['gamma_correction'] = [True, [.85, 1.15]]                                    #whether to apply gamma correction to input channels, and range for gamma factor
params_dict['deform_mesh'] = [True, [10.0, 50.0], 3.5, 3]                                #whether to apply deformation augmentation, scale factors for normal/abnormal, sigma of gaussian smoother, truncate filter sigma, spline order for interpolation of volume and label 
params_dict['scale_mesh'] = [True, [.75, 1.35], False]                                   #whether to apply scaling augmentation, and the bounds for scaling operation (less than 1 zooms in; greater than 1 zooms out); whether scaling should be isotropic or not
params_dict['rotate_mesh'] = [True, np.array([-20, 20]) * np.pi / 180]                   #whether to apply rotation augmentation, and the bounds for rotating operation (in radians)
params_dict['shear_mesh'] = [True, [-.15, .15], False]                                   #whether to apply shear augmentation, and the bounds for shear operation (values close to zero will produce sensible shears); whether to shear only along one direction, or to allow shearing along all valid axes
params_dict['translate_mesh'] = [True, [-3, 3]]                                          #whether to apply translation augmentation, and the bounds for translating operation (in voxels)
params_dict['erode_dilate_conn_comp'] = [[False, .1, 3], [False, .1, 3], [False, .1]]    #only applicable if passing in binary mask as input; whether to use erosion augmentation, probability of erosion, and max number of iterations; whether to use dilation augmentation, probability of dilation, and max number of iterations; whether to delete connected components at random, and probability of deletion
params_dict['randomize_order_augmentations'] = True                                      #whether to randomize the order of composable affine transforms (scale, rotate, shear) or to use the default order every time
params_dict['spline_order'] = [3, 0, 1]                         #spline order for applying deformations to volumes and label maps (and distance maps if applicable)
params_dict['num_epochs'] = 10000                               #number of epochs to train the model
params_dict['early_stopping_threshold'] = [2500, 100, 5.]       #number of epochs before breaking from training loop if no improvement in validation monitor, number of epochs to drop learning rate (if not using LR schedule), factor by which to drop learning rate
params_dict['verbose'] = 2                                      #0 shows no output, 1 shows training progress bar (not recommended for non-interactive jobs), 2 shows one line per epoch (also not recommended since have callbacks enabled)
params_dict['workers'] = 10                                     #number of parallel processes to generate to handle data loading
params_dict['max_queue_size'] = 10                              #maximum number of entries in the generator queue
params_dict['cache_volumes'] = True                             #whether to cache the volumes/labels into a dictionary (recommended if have enough memory to store entire dataset in memory)
params_dict['enable_xla'] = False                                #whether to enable xla compilation (will see performance boost, but compilation will fail unless carefully construct network to utilize only multiples of 2 and 8 everywhere)
params_dict['mixed_precision'] = True                            #whether to use mixed_precision
params_dict['custom_fit'] = [False, False, [False, 2.], False]   #whether to override model.fit; whether to apply weight decay to all variables (or just apply it to kernels, leaving bias/scale un-regularized); whether to increase learning rate of bias/scale terms to speed up convergence of those terms; whether to use gradient centralization
#optional custom learning rate schedule parameters
params_dict['lr_schedule_type'] = 'CosineAnneal'                        #learning rate schedule to use (choose from 'None' (which will default to reducing learning rate on plateau), 'StepDecay', 'CosineAnneal', 'Exponential')
params_dict['use_lr_warmup'] = [True, 5, 1000]                          #whether to use learning rate warmup, how many epochs to warmup for, factor smaller than actual learning rate to start warmup period at
#params for optional step decay learning rate decay schedule
params_dict['step_epoch_nums'] = [30, 50, 70, 85]                       #epochs at which learning rate decay will occur (make sure to take into account learning rate warmup if using that)
params_dict['step_factors'] = [1, 5, 25, 125, 625]                      #factor by how much initial learning rate should be decayed at corresponding steps (must be list of length one greater than length of 'step_epoch_nums')
#params for optional cosine annealing learning rate decay schedule (params shared with exponential schedule as well)
params_dict['cycle'] = True                                             #whether to make schedule cyclic
params_dict['num_epochs_cycle'] = 250                                   #number of epochs in the first cycle
params_dict['factor_decrease_learning_rate'] = 250                      #amount to decrease learning rate by the end of the cycle
params_dict['cycle_multiplier'] = 1.                                    #how much to increase cycle length with each new cycle 
params_dict['factor_decrease_max_rate_per_cycle'] = 1.                  #how much to decay max learning rate at start of new cycle (closer to 1 decays less)
params_dict['factor_decrease_min_rate_per_cycle'] = 1.                  #how much to decay min learning rate at start of new cycle (closer to 1 decays less)
params_dict['save_at_cycle_end'] = False                                #whether to save the model at the end of the cycle
#params for optional exponential learning rate decay schedule
params_dict['power'] = 0.15                                             #power to reduce rate (closer to 1 makes function more linear; closer to 0 makes function more like a step function)
#predict params 
params_dict['predict_using_patches'] = True                                             #whether to predict patchwise or pass entire volume into convolutional net and predict in a single pass
params_dict['patch_overlap'] = 0.95                                                     #precentage of overlap with neighboring patches (i.e. .75 means 75% overlap)
params_dict['percentage_empty_to_skip_patch'] = [0.5, True]                             #patches at the borders of the image will be mostly empty and are not worth predicting on. This parameter skips patches if they are below the threshold selected (i.e. 0.5 means that patches that are 50% empty will be skipped); whether you want to predict a patch if the center of the patch contains data (regardless of what percentage of the patch is actually empty)
params_dict['boundary_removal'] = 2                                                     #number of voxels to remove around the edge of the predicted patch to combat edge effects
params_dict['blend_windows_mode'] = ['constant']                                        #how to blend sliding window patches together (choose between 'constant', 'linear', or 'gaussian'). 'constant' gives equal weight to all predictions; 'linear' gives full weight to middle of patch and linearly decreases weight of samples close to edge; 'gaussian' gives less weight to predictions on edges of windows. If using 'gaussian', provide sigma(s) to determine kernel weights along spatial dimensions (recommended setting between 0.15 and 0.35)
params_dict['average_logits'] = True                                                    #whether raw logits should be averaged together during patch-based inference and sigmoid probability taken at very end
params_dict['logits_clip_value'] = 50                                                   #what value above/below to clip logits to prevent overflow errors in sigmoid function
params_dict['predict_left_right_patch'] = [False, params_dict['left_right_flip'][1]]    #whether to predict on both the left and right flipped patch and then average results, and axis along which to perform flip
params_dict['predict_multi_scale'] = [False, params_dict['scale_mesh'][1], 3]           #whether to predict on scaled inputs (not implemented for patchwise inference); what scales to compute predictions over; what order interpolation to use (anything other than order 0 or order 1 will take a very long time to run)
params_dict['binarize_value'] = [0.5]                                                   #probability at which to threshold sigmoid function outputs at   
params_dict['number_snapshots_ensemble'] = [0]                                          #which snapshots to ensemble together (must be list with no more than "max_number_checkpoints_keep" values)
params_dict['predict_fast_to_find_roi'] = [False, .95, [-1,-1,-1]]                      #whether to first run inference using small patch_overlap and then predict again using the true patch overlap only on the region of interest that has been detected; how many slices above/below ROI to add in case small patch_overlap is incorrect
params_dict['save_uncertainty_map'] = [False, 'uncertainty-mask.nii.gz']                #whether to create uncertainty mask from different predictions; name of uncertainty mask volume
#unet params
params_dict['network_name'] = 'ClassificationNetwork'                                   #name of network to use during training (choose between 'Unet' and 'ClassificationNetwork'); 'ClassificationNetwork' can be used as a generic ResNet/DenseNet by modulating the options below
params_dict['block_type'] = 'Dense_Block'                                               #block type to use in unet (choose between 'Convolutional_Block', 'Dense_Block', 'Residual_Block', 'Residual_Bottleneck_Block')
params_dict['preactivation'] = [False, 32, 7]                                           #whether to use pre-activated blocks (Norm->Act->Conv) instead of post-activated blocks (Conv->Norm->Act); maximum number of filters to use in first filter; kernel size for first convolution
params_dict['levels'] = 4                                                               #number of levels in u-net
params_dict['atrous_conv'] = [0, np.array([2,2,2])]                                     #number of levels (from the bottom) that use atrous convolutions with no pooling or skip connections (max=levels-1), and size of initial dilation
params_dict['filter_num_per_level'] = [256,512,1024,2048]                                   #number of filters in each layer of the network (must be list of length equal to levels parameter and each successive value must be double the previous or you might run into compilation errors on some of the more complex unets)
params_dict['blocks_per_level'] = 1                                                     #number of blocks (i.e. conv->norm->act) per level; if an integer, uses that number of blocks on each level throughout the network, else input a list of length 2*levels-1 with number of blocks on each level (useful if want to create asymmetrical encoder-decoder)
params_dict['dense_growth_rate_per_level'] = 32                                         #growth rate if using dense block; if an integer, uses that same growth rate on each level throughout the network, else input a list of length 2*levels-1 with growth rate of each level
params_dict['dense_bottleneck'] = [True, 4, False]                                      #whether to use bottleneck layers in dense block; how much larger than the growth rate the bottleneck size should be; whether to only use bottleneck layers if the current number of filters is less than or equal to the size of the bottleneck layer
params_dict['deep_supervision'] = [True, 2]                                             #whether to use deep supervision, and the number of intermediary levels to use it with
params_dict['num_inputs'] = len(params_dict['input_image_names'])                       #number of inputs to network
params_dict['num_outputs'] = 1                                                          #number of output channels to predict
params_dict['activation'] = ['ReLU']                                                    #activation function (case-sensitive) (choose from 'ReLU', 'LeakyReLU')
params_dict['filter_size_conv'] = 3                                                     #size of convolutional kernel (can specify as tuple if want non-uniform sized kernel i.e. (5,5,3))
params_dict['filter_size_conv_transpose_pool'] = (2,2,2)                                #size of transposed convolutional / max pooling kernel (change accordingly if using thick slice patches (i.e. size (2,2,1) for patch size [128,128,32]))
params_dict['num_repeat_pooling'] = [-1,-1,1]                                           #number of times to repeat pooling (if -1, then pools as many times as there are levels in the network, else pools only specified number of times along axis); For example, a 5 level network with input patch size of [128, 128, 32] and num_repeat_pooling of [-1, -1, 2] would pool as follows: [128,128,32]->[64,64,16]->[32,32,8]->[16,16,8]->[8,8,8] 
params_dict['padding'] = 'same'                                                         #type of padding for convolutional operations (highly suggested to leave as 'same')
params_dict['use_bias_on_convolutions'] = False                                         #whether to use bias term on convolution (not necessary if applying normalization)
params_dict['use_scale_on_normalization'] = False                                       #whether to use scale term on batch normalization (not necessary if applying linear activation (or relu))
params_dict['use_grouped_convolutions'] = [False, 4, 2, False]                          #whether to use grouped convolutions, minimum group size, maximum cardinality, whether to use native grouped convolution or naive loop based implementation (loop based implementation seems to train better (something with the gradients/updates in the optimized version))
params_dict['use_selective_kernel'] = [False, 2, 8, 16]                                 #whether to use selective kernel block, number of branches, reduction ratio for bottleneck, minimum size of bottleneck (highly recommended to use selective kernels with grouped convolution to keep parameter count low (even though this will substantially increase training time))
params_dict['use_cbam'] = [False, 8, 16]                                                #whether to use convolutional block attention module, reduction ratio for bottleneck, minimum size of bottleneck
params_dict['use_attentive_normalization'] = [False, 8, 32, [10,10,15,20]]              #whether to use attentive normalization, reduction ratio for bottleneck, minimum size of bottleneck, value of K at each layer of the network
params_dict['loss'] = 'binary_cross_entropy_classification'                             #loss function (from loss_functions.py file)
params_dict['metrics'] = ['binary_accuracy', 'binary_accuracy_batch']                   #metrics (from loss_functions.py file)
params_dict['monitor'] = ['binary_accuracy_batch', 'max']                               #monitor for saving model, and direction ('min' or 'max') that indicates better model performance 
params_dict['optimizer'] = ['SGDW', params_dict['weight_decay'], params_dict['learning_rate'], .9]   #optimizer along with parameters (i.e. SGDW with weight decay, learning rate, and momentum)
params_dict['pooling'] = 'MaxPool'                                                      #pooling operation (choose from 'MaxPool', 'AveragePooling', 'StridedConvolution', 'ProjectionShortcut' (ProjectionShortcut only usable with residual blocks and cannot be used with atrous convolutions or selective kernels since tensorflow does not support convolutions that are both strided and dilated))
params_dict['anti_aliased_pooling'] = [False, False]                                    #whether to use anti-aliased versions of MaxPool and StridedConvolution via the BlurPool operator; whether to use large 4x4x4 kernel or small 2x2x2 box kernel
params_dict['upsampling'] = ['Trilinear', True]                                         #type of upsampling to use in decoder (choose from 'Deconvolution' or 'Trilinear'); whether to use grouped convolution or sparse weight matrix for upsampling (uses more memory/flops but might actually be slightly faster in practice)
params_dict['normalization'] = ['BatchNormalization', True]                             #normalization (choose from 'BatchNormalization', 'LayerNormalization', 'GroupNormalization', 'InstanceNormalization'); if using BatchNormalization, specify whether want to train batch statistics; if using GroupNormalization, specify number of channels per group and a minimum number of groups (both must be able to evenly divide filter_num_per_level parameter)
params_dict['regularization_values'] = [0.0, 0.0]                                       #regularization coefficients for L1 and L2 kernal regularizers
params_dict['weight_standardization'] = False                                           #whether to apply weight standardization to all convolution kernels that come before a normalization layer (i.e. force kernel to be zero mean unit variance)
params_dict['dropblock_params'] = None #[None, None, None, [.8,8,[16,16,8]], [.8,8,[8,8,8]]]        #probability for dropblock at each level of network and block size at that level
params_dict['dropblock_scheduler'] = [False, 125]                                       #whether to use scheduling for dropblock, and the number of epochs to reach full weighting
params_dict['conv_weights_initializer'] = 'he_normal'                                   #initializer for convolutional layers (choose from 'he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform')
params_dict['retina_net_bias_initializer'] = [True, 0.33]                               #whether to use a retina-net style initializer for the bias of the final convolutional layer, and the a priori probability of the foreground class (will help limit massive gradient that occurs due to incorrectly classified background examples)
params_dict['zero_init_for_residual'] = False                                           #whether to set the scale term on the normalization layer to zero at initialization to ensure all gradient flow through the identity connection at the start of training
#general loss function params
params_dict['factor_reweight_foreground_classes'] = [1., 4.]      #list of size equal to the number of classes (including background class) (i.e. list of size two for binary task, list of size 4 for task with 3 foreground classes and 1 background, etc) with factor to change weighting of each class in cross entropy losses
params_dict['gamma'] = 2.                                         #gamma power term for focal loss
params_dict['dice_over_batch'] = True                             #whether to treat the batch as a pseudo-volume and calcuate dice over the entire matrix, or calculate dice over each patch individually and average
params_dict['dice_with_background_class'] = False                 #whether to use the background class dice in the loss function
params_dict['use_sigmoid_for_multi_class'] = False                #if doing multi-class segmentation, whether to use softmax to generate probabilities or use sigmoid to generate (overlapping) class probabilities
params_dict['joint_loss_function_params'] = [1.0, 1.0, False]     #if using joint loss function, 1) initial loss function weighting, 2) decay factor to change weighting over time, 3) whether to apply inverse decay (i.e. weighting for loss function one gets reduced over time while weighting for loss function two gets increased over time)

#make output directory if it does not already exist
if not os.path.exists(params_dict['model_outputs_dir']):
    os.makedirs(params_dict['model_outputs_dir'])

#run training or prediction using given config
if trainModel == True:
    train_model(params_dict)
else:
    predict_model(params_dict)