#TODO: If re-starting training for a certain saved model, evaluate validation set first to get accurate monitor value
#TODO: If re-starting training, ensure original model isn't overwritten
#TODO: Add ability to choose model to use for training/inference from snapshots
#TODO: Fix random seed issue to make training fully deterministic
#TODO: Allow spreadsheet of train/val cases for easy cross-validation without needing to split cases?

import os
import random
import numpy as np
import tensorflow as tf

from model_callbacks import *
from sorcery import unpack_keys
from unet import create_model, load_my_model
from load_data import DataGenerator, nested_folder_filepaths

def train_model(params_dict):
    #unpack relevant dictionary elements
    random_seed, reload_model, output_file, dropblock_scheduler, tensorboard_dir, lr_schedule_type, loss, joint_loss_function_params, data_dir_train, data_dir_val, num_patches_per_patient, adaptive_full_image_patching, input_image_names, ground_truth_label_names, num_epochs, verbose, workers, max_queue_size = unpack_keys(params_dict)
    #set random seed if desired
    if random_seed[0] == True:
        seed_value = random_seed[1]
        os.environ['PYTHONHASHSEED']=str(seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
    else:
        np.random.seed()
    #enable multi-gpu training set-up if more than one GPU is visible
    strategy = tf.distribute.MirroredStrategy()
    numGPUs_available = strategy.num_replicas_in_sync
    with open(output_file, 'a') as f:
        f.write('Training on ' + str(numGPUs_available) + ' GPUs simultaneously \n')
    params_dict['batch_size'][0] = params_dict['batch_size'][0] * numGPUs_available
    params_dict['batch_size'][1] = params_dict['batch_size'][1] * numGPUs_available
    #get patients in train and val sets
    train_patients = nested_folder_filepaths(data_dir_train, [input_image_names, ground_truth_label_names])
    val_patients = nested_folder_filepaths(data_dir_val, [input_image_names, ground_truth_label_names])
    iterations_per_epoch = np.ceil((len(train_patients) * num_patches_per_patient[0]) / params_dict['batch_size'][0])
    #instantiate callbacks list
    callbacks = []
    #if using dropblock with scheduling, convert desired number of epochs to iterations
    if dropblock_scheduler[0] == True:
        params_dict['dropblock_scheduler'][1] = dropblock_scheduler[1] * iterations_per_epoch
        #also add scheduling callback
        DropblockScheduleCallback = DecayDropblockProbability()
        callbacks.append(DropblockScheduleCallback)
    #load/compile model
    if numGPUs_available > 1:
        with strategy.scope():
            if reload_model[0] == False:
                model, manager = create_model(params_dict)
            else:
                model, manager, _ = load_my_model(params_dict, train_mode=True)
    else:
        if reload_model[0] == False:
            model, manager = create_model(params_dict)
        else:
            model, manager, _ = load_my_model(params_dict, train_mode=True)
    #print model summary to output file
    with open(output_file, 'a') as f:
        model.summary(line_length=150, print_fn=lambda x: f.write(x + '\n'))
    #load tensorboard callback and specific learning rate logger
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_graph=True, histogram_freq=1, profile_batch=0)
    file_writer = tf.summary.create_file_writer(tensorboard_dir + '/learning_rate')
    file_writer.set_as_default()
    #load callback to print and output information to text file
    logging_callback = PrintLossAndMetric(params_dict)
    #load early stopping / learning rate reduction on plateau callback
    early_stopper_callback = EarlyStoppingAndReduceOnPlateau(manager, params_dict)
    callbacks.extend([logging_callback, early_stopper_callback, tensorboard_callback])
    #load in learning rate schedule if specified
    if lr_schedule_type != 'None':
        learning_rate_schedule_callback = CustomLearningRateSchedules(params_dict, iterations_per_epoch)
        callbacks.append(learning_rate_schedule_callback)
    #if chosen loss function is a joint loss, initialize alpha parameter callback
    if 'joint' in loss:
        AlphaCallback = DecayAlphaParameter(*joint_loss_function_params)
        callbacks.append(AlphaCallback)
    #load data generators (shuffle training data; no need to shuffle validation data)
    train_generator = DataGenerator(data_dir_train, train_patients, params_dict['batch_size'][0], num_patches_per_patient[0], adaptive_full_image_patching[0], 'train', params_dict)
    val_generator = DataGenerator(data_dir_val, val_patients, params_dict['batch_size'][1], num_patches_per_patient[1], adaptive_full_image_patching[1], 'val', params_dict)
    #train model for desired number of epochs or until early stopping criteria met (shuffling turned off since our data loader shuffles every epoch)
    history = model.fit(x=train_generator, epochs=num_epochs, validation_data=val_generator, callbacks=callbacks, workers=workers, max_queue_size=max_queue_size, verbose=verbose, shuffle=False, use_multiprocessing=False)