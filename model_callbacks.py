#TODO: Normalize weight decay to cycle length
#TODO: Combine early stopping / reduce on plateau / custom learning rate schedules into one callback
#TODO: Save validation monitor value in model checkpoint save name

import time
import pickle
import numpy as np
import tensorflow as tf

from sorcery import unpack_keys

#callback to control cyclic learning rate schedule
class CustomLearningRateSchedules(tf.keras.callbacks.Callback):
    def __init__(self, params_dict, iterations_per_epoch):
        super(CustomLearningRateSchedules, self).__init__()
        #unpack relevant dictionary elements
        lr_schedule_type, learning_rate, weight_decay, use_lr_warmup, step_epoch_nums, step_factors, cycle, num_epochs_cycle, factor_decrease_learning_rate, cycle_multiplier, factor_decrease_max_rate_per_cycle, factor_decrease_min_rate_per_cycle, power = unpack_keys(params_dict)
        self.iterations_per_epoch = iterations_per_epoch
        self.lr_schedule_type = lr_schedule_type
        self.initial_learning_rate = learning_rate
        self.initial_weight_decay = weight_decay
        self.current_iteration = 0
        self.current_epoch = 0
        #params for learning rate warmup
        self.use_lr_warmup = use_lr_warmup
        self.min_lr_warmup = self.initial_learning_rate / self.use_lr_warmup[2]
        self.min_wd_warmup = self.initial_weight_decay / self.use_lr_warmup[2]
        self.current_warmup_step = 0.0
        #params for step decay
        self.step_epoch_nums = np.array(step_epoch_nums)
        self.step_factors = step_factors
        #params for cosine annealing learning rate decay (and exponential)
        self.cycle = cycle
        self.num_epochs_cycle = num_epochs_cycle
        self.factor_decrease_learning_rate = factor_decrease_learning_rate
        self.cycle_multiplier = cycle_multiplier
        self.factor_decrease_max_rate_per_cycle = factor_decrease_max_rate_per_cycle
        self.factor_decrease_min_rate_per_cycle = factor_decrease_min_rate_per_cycle
        self.power = power
        self.min_lr = self.initial_learning_rate / self.factor_decrease_learning_rate
        self.min_wd= self.initial_weight_decay / self.factor_decrease_learning_rate
        self.current_step = 0.0
    #
    def on_train_batch_begin(self, batch, logs=None):
        if self.use_lr_warmup[0] == True and self.current_epoch < self.use_lr_warmup[1]:
            self.update_warmup()
        elif self.lr_schedule_type == 'StepDecay':
            self.update_step_decay()
        elif self.lr_schedule_type == 'CosineAnneal':
            self.update_cosine_anneal()
        elif self.lr_schedule_type == 'Exponential':
            self.update_exponential()
        #update global iteration counter
        self.current_iteration = self.current_iteration + 1
        #update global epoch counter
        if self.current_iteration % self.iterations_per_epoch == 0:
            self.current_epoch = self.current_epoch + 1
    #
    def update_warmup(self):
        current_position = self.current_warmup_step / (self.iterations_per_epoch * self.use_lr_warmup[1])
        new_lr_warmup = self.min_lr_warmup + (self.initial_learning_rate - self.min_lr_warmup) * current_position
        if hasattr(self.model.optimizer, 'weight_decay') and tf.is_tensor(self.model.optimizer.weight_decay):
            new_wd_warmup = self.min_wd_warmup + (self.initial_weight_decay - self.min_wd_warmup) * current_position
        else:
            new_wd_warmup = None
        self.set_learning_rate_and_weight_decay(new_lr_warmup, new_wd_warmup)
        self.current_warmup_step = self.current_warmup_step + 1
    #
    def update_step_decay(self):
        current_step_decay = np.where(self.current_epoch < self.step_epoch_nums)[0]
        if np.any(current_step_decay):
            factor_reduce_index = current_step_decay[0]
        else:
            factor_reduce_index = -1
        new_lr_step_decay = self.initial_learning_rate / self.step_factors[factor_reduce_index]
        if hasattr(self.model.optimizer, 'weight_decay') and tf.is_tensor(self.model.optimizer.weight_decay):
            new_wd_step_decay = self.initial_weight_decay / self.step_factors[factor_reduce_index]
        else:
            new_wd_step_decay = None
        self.set_learning_rate_and_weight_decay(new_lr_step_decay, new_wd_step_decay)
    #
    def update_cosine_anneal(self):
        current_position = self.update_general()
        new_lr_cosine = self.min_lr + 0.5 * (self.initial_learning_rate - self.min_lr) * (1 + np.cos(current_position * np.pi))
        if hasattr(self.model.optimizer, 'weight_decay') and tf.is_tensor(self.model.optimizer.weight_decay):
            new_wd_cosine = self.min_wd + 0.5 * (self.initial_weight_decay - self.min_wd) * (1 + np.cos(current_position * np.pi))
        else:
            new_wd_cosine = None
        self.set_learning_rate_and_weight_decay(new_lr_cosine, new_wd_cosine)
    #
    def update_exponential(self):
        current_position = self.update_general()
        new_lr_exponential = self.min_lr + (self.initial_learning_rate - self.min_lr) * ((1.0 - current_position) ** self.power)
        if hasattr(self.model.optimizer, 'weight_decay') and tf.is_tensor(self.model.optimizer.weight_decay):
            new_wd_exponential = self.min_wd + (self.initial_weight_decay - self.min_wd) * ((1.0 - current_position) ** self.power)
        else:
            new_wd_exponential = None
        self.set_learning_rate_and_weight_decay(new_lr_exponential, new_wd_exponential)
    #
    def update_general(self):
        current_position = self.current_step / (self.iterations_per_epoch * self.num_epochs_cycle)
        if self.cycle == True:
            if current_position == 1.0:
                #end of cycle
                self.current_step = 0
                self.num_epochs_cycle = self.num_epochs_cycle * self.cycle_multiplier
                self.initial_learning_rate = self.initial_learning_rate / self.factor_decrease_max_rate_per_cycle
                self.min_lr = self.min_lr / self.factor_decrease_min_rate_per_cycle
                self.initial_weight_decay = self.initial_weight_decay / self.factor_decrease_max_rate_per_cycle
                self.min_wd = self.min_wd / self.factor_decrease_min_rate_per_cycle
        else:
            current_position = min(current_position, 1)
        self.current_step = self.current_step + 1
        return current_position
    #
    def set_learning_rate_and_weight_decay(self, new_lr, new_wd=None):
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
        if new_wd != None:
            tf.keras.backend.set_value(self.model.optimizer.weight_decay, new_wd)
        tf.summary.scalar('learning_rate', data=new_lr, step=self.current_iteration)
        
#callback to print loss and metrics to text file
class PrintLossAndMetric(tf.keras.callbacks.Callback):
    def __init__(self, params_dict):
        super(PrintLossAndMetric, self).__init__()
        #unpack relevant dictionary elements
        loss, metrics, output_file = unpack_keys(params_dict)
        self.loss_name = loss
        self.metrics_name_list = metrics
        self.output_file = output_file
        self.start_epoch = 0.0
    #
    def on_epoch_begin(self, epoch, logs=None):
        self.start_epoch = time.time()
    #
    def on_epoch_end(self, epoch, logs=None):
        timeForEpoch = int(time.time() - self.start_epoch)
        epoch_summary_str = 'Epoch ' + str(epoch) + ': ' + str(timeForEpoch) + 's - train ' + self.loss_name + ' = ' + str(logs['loss']) + ', val ' + self.loss_name + ' = ' + str(logs['val_loss'])
        for metric in self.metrics_name_list:
            epoch_summary_str = epoch_summary_str + ', train ' + metric + ' = ' + str(logs[metric]) + ', val ' + metric + ' = ' + str(logs['val_' + metric])
        with open(self.output_file, 'a') as f:
            f.write(epoch_summary_str + '\n')

#callback to reduce learning rate over time at validation monitor plateau and to stop training after significant plateau
class EarlyStoppingAndReduceOnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, manager, params_dict):
        super(EarlyStoppingAndReduceOnPlateau, self).__init__()
        #unpack relevant dictionary elements
        model_config_save_path, monitor, early_stopping_threshold, lr_schedule_type, learning_rate, weight_decay, save_at_cycle_end, use_lr_warmup, num_epochs_cycle, output_file = unpack_keys(params_dict)
        self.manager = manager
        self.model_config_save_path = model_config_save_path
        self.monitor = monitor[0]
        self.mode = monitor[1]
        if self.mode == 'max':
            self.monitor_value = -np.Inf
        else:
            self.monitor_value = np.Inf
        self.loss_value = np.Inf
        self.early_stopping_threshold = early_stopping_threshold
        self.lr_schedule_type = lr_schedule_type
        self.initial_learning_rate = learning_rate
        self.initial_weight_decay = weight_decay
        self.save_at_cycle_end = save_at_cycle_end
        if self.save_at_cycle_end == True:
            self.num_epochs_cycle = num_epochs_cycle
            if use_lr_warmup[0] == True:
                self.lr_warmup_num = use_lr_warmup[1]
            else:
                self.lr_warmup_num = 0
        self.output_file = output_file
        self.early_stopping_counter = 0
        self.epoch_save = 0
        self.early_stop = False
    #
    def on_epoch_end(self, epoch, logs=None):
        current = logs['val_' + self.monitor]
        current_loss = logs['val_loss']
        if self.save_at_cycle_end == True:
            if (epoch + 1) > self.lr_warmup_num and (epoch + 1 - self.lr_warmup_num) % self.num_epochs_cycle == 0:
                self.epoch_save = epoch
                self.save_my_model()
        #if validation monitor value has improved, save model; if validation monitor value is the same, but validation loss has improved, save the model
        if (self.mode == 'max' and (current > self.monitor_value or (current == self.monitor_value and current_loss < self.loss_value))) or (self.mode == 'min' and (current < self.monitor_value or (current == self.monitor_value and current_loss < self.loss_value))):
            self.monitor_value = current
            self.loss_value = current_loss
            self.early_stopping_counter = 0
            self.epoch_save = epoch
            self.save_my_model()
        else:
            self.early_stopping_counter = self.early_stopping_counter + 1
            #if have not improved monitor value after sufficiently long, stop training
            if self.early_stopping_counter >= self.early_stopping_threshold[0]:
                self.model.stop_training = True
                self.early_stop = True
            if self.lr_schedule_type == 'None':
                #if validation metric has plateaued for significant portion of time, drop learning rate (and weight decay) by requested factor
                if self.early_stopping_counter % self.early_stopping_threshold[1] == 0:
                    self.initial_learning_rate = self.initial_learning_rate / self.early_stopping_threshold[2]
                    tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.initial_learning_rate)
                    if hasattr(self.model.optimizer, 'weight_decay') and tf.is_tensor(self.model.optimizer.weight_decay):
                        self.initial_weight_decay = self.initial_weight_decay / self.early_stopping_threshold[2]
                        tf.keras.backend.set_value(self.model.optimizer.weight_decay, self.initial_weight_decay)          
        if self.lr_schedule_type == 'None':
            tf.summary.scalar('learning_rate', data=self.initial_learning_rate, step=epoch)
    #
    def on_train_end(self, logs=None):
        if self.early_stop == True:
            with open(self.output_file, 'a') as f:
                f.write('Training stopped early due to no performance gain in validation monitor! \n')
        else:
            with open(self.output_file, 'a') as f:
                f.write('Training completed for specified number of epochs! \n')
        with open(self.output_file, 'a') as f:
            f.write('Last model saved at epoch ' + str(self.epoch_save) + ' with best validation ' + self.monitor + ' of ' + str(self.monitor_value) + ' \n')
    #
    def save_my_model(self):
        #save config of Unet class (which is first layer of model after placeholder input layer)
        with open(self.model_config_save_path, 'wb') as f:
            pickle.dump(self.model.layers[1].get_config(), f)
        #save model weights using tensorflow checkpoint format
        self.manager.save(checkpoint_number=self.epoch_save)
        with open(self.output_file, 'a') as f:
            f.write('Model saved! \n')
        tf.print('Model saved with validation ' + self.monitor + ' of ' + str(self.monitor_value) + '!')

#callback to decay class variable alpha in joint loss functions
class DecayAlphaParameter(tf.keras.callbacks.Callback):
    alpha1 = tf.keras.backend.variable(1.0)
    alpha1._trainable = False
    alpha2 = tf.keras.backend.variable(1.0)
    alpha2._trainable = False
    def __init__(self, alpha1_initialization=0.99, decay_factor=0.925, inverse_decay=False):
        super(DecayAlphaParameter, self).__init__()
        self.alpha1_initialization = alpha1_initialization
        self.decay_factor = decay_factor
        self.inverse_decay = inverse_decay
        tf.keras.backend.set_value(DecayAlphaParameter.alpha1, self.alpha1_initialization)
        if self.inverse_decay == False:
            tf.keras.backend.set_value(DecayAlphaParameter.alpha2, self.alpha1_initialization)
        else:
            tf.keras.backend.set_value(DecayAlphaParameter.alpha2, 1 - self.alpha1_initialization)
    #
    def on_epoch_end(self, epoch, logs=None):
        tf.keras.backend.set_value(DecayAlphaParameter.alpha1, self.alpha1_initialization * (self.decay_factor ** epoch))
        if self.inverse_decay == False:
            tf.keras.backend.set_value(DecayAlphaParameter.alpha2, DecayAlphaParameter.alpha1)
        else:
            tf.keras.backend.set_value(DecayAlphaParameter.alpha2, 1 - DecayAlphaParameter.alpha1)

#callback to decay dropblock probability
class DecayDropblockProbability(tf.keras.callbacks.Callback):
    def __init__(self):
        self.intermediate_block_names = ['encoder', 'decoder', 'encoding_convolutional_layer', 'decoding_convolutional_layer']
        self.terminal_block_names = ['convolutional_block', 'residual_block', 'residual_bottleneck_block', 'dense_block']
        self.terminal_blocks = []
    #
    def on_train_begin(self, logs=None):
        #start at unet source node and recursively search for all blocks that contain dropblock
        self.recursive_layer_search(self.model.layers[1])
    #    
    def on_train_batch_end(self, batch, logs=None):
        #decay each block keep_prob variable as per scheduled routine
        for block in self.terminal_blocks:
            if block.dropblock_params != None:
                current_keep_prob = tf.keras.backend.get_value(block.current_prob)
                new_keep_prob = max(current_keep_prob - block.amount_decrement, block.final_prob)
                tf.keras.backend.set_value(block.current_prob, new_keep_prob)
    #
    def recursive_layer_search(self, current_layer):
        for layer in current_layer.layers:
            if layer.name in self.intermediate_block_names:
                self.recursive_layer_search(layer)
            elif layer.name in self.terminal_block_names:
                self.terminal_blocks.append(layer)

'''
# Code to clean up tf_ckpts directory, especially when fine-tuning a previously trained model
import os
checkpoint_dir = 'model_outputs/tf_ckpts/'
f = open(checkpoint_dir + 'checkpoint', "r")
checkpoints_to_keep = []
for columns in (raw.strip().split() for raw in f):
    if columns[0] == 'all_model_checkpoint_paths:':
        checkpoints_to_keep.append(columns[1][1:-1] + '.')

checkpoint_files = next(os.walk(checkpoint_dir))[2]
for checkpoint_file in checkpoint_files:
    if checkpoint_file != 'checkpoint' and not any([checkpoint_to_keep in checkpoint_file for checkpoint_to_keep in checkpoints_to_keep]):
        os.remove(checkpoint_dir + checkpoint_file)
'''