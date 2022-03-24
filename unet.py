#TODO: Why does optimized grouped convolution not train as well as loop based implementation
#TODO: XLA compilation does not work with symmetric tf.pad (but it works fine with zero-padding) -> replace tf.pad with some layer that extends boundary manually?
#TODO: Use symmetric padding for upsampling; add option to use different padding on all conv filters
#TODO: Figure out how to use grouped convolutions effectively (i.e. start by drastically reducing group number so that still get memory reduction but also see if it works with SK)
#TODO: Channels first is slightly faster than channels last (about 2% faster on average) -> will need to change lots of things to make this work
#TODO: Figure out how to reuse kernels for selective kernel block
#TODO: add option to not use skip connections (i.e. no long range concatenation)
#TODO: unet++ if feeling bored (would not be terribly difficult to add, just need to figure out best way to add to loss function)
#TODO: add projection shortcut to standard conv block (i.e. strided conv in first conv filter in the block instead of outside the block)
#TODO: Previous optimizer state is always loaded from previous model -> look into how to load just the weights of the model with a fresh optimizer state
#TODO: Figure out failing to serialize error
#TODO: Try joint max/average pool
#TODO: Figure out why zero init for residual is slowing down training so much and not seemingly working (seems like scale term is not shifting much away from 0 (and is mostly very small negative), thus making each residual branch learn a very weak function -> might need to boost learning rate of just these terms to be useful?)
#TODO: figure out better way to write blocks so that there is not so much copy/pasted code everywhere
#TODO: figure out what "regularization_losses" is for custom fit loss
#TODO: is there a way to see tensor shape at run-time?
#TODO: fix load_my_model to better use saved information while incorporating new info (patch size, regularization, data augmentation, etc.)
#TODO: Need to modify custom training loop to handle multi-gpu set-up
#TODO: Add SyncBatchNormalization as option (and compare performance against normal BatchNormalization when applied in distributed setting)
#TODO: add option for multiple cascaded u-nets (and furthermore, allow first unet to use downsampled input image for anatomical context)
#TODO: finish attentive normalization code (add RSD and optional batch norm of instance specific lambdas)
#TODO: add option to have different blocks at different levels of the network
#TODO: remove num_groups option from instantiate normalization layer (deprecated)
#TODO: convert everything to pre-act (fix first conv input / convert residual blocks to pre-act)
#TODO: why does preactivated networks train slower than non-preactivated networks and why do they overfit faster?
#TODO: shift-invariant pooling/upsampling (remove hard-coding for filter size 4 and for 3D inputs)
#TODO: Finish transformations to allow for more accurate plotting of neural network architectures (proper shape and stride information from tf_builder.py script?)
#TODO: check if gradient centralization helps
#TODO: check if regularizing biases/scales helps
#TODO: Don't force doubling of filters at each level in case want to max saturate filter num
#TODO: Diagnose issue with using preactived dense blocks during training of dense net classifier

import os
import tensorflow as tf
import tensorflow_addons.layers as tfa_layers
import tensorflow_addons.optimizers as tfa_optimizers

import pickle
import numpy as np
import loss_functions
import hiddenlayer as hl
import hiddenlayer.transforms as ht

from copy import deepcopy
from sorcery import unpack_keys

#standard residual block (computes [[Conv3->Norm->Act->Conv3->Norm]+shortcut]->Act)
class Residual_Block(tf.keras.Model):
    def __init__(self, num_filters, dilation_rate, operations_dictionary, params_dict, in_filters=None, pool=None, dropblock_params=None, dense_parameters=None, attentive_normalization_K=None, name='residual_block', **kwargs):
        super(Residual_Block, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.num_filters = num_filters
        self.dilation_rate = dilation_rate
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        if in_filters == None:
            self.in_filters = num_filters
        else:
            self.in_filters = in_filters
        if pool == None:
            self.pool = None
        else:
            self.pool = deepcopy(pool)
        #unpack relevant dictionary elements
        use_grouped_convolutions, use_selective_kernel, use_cbam, use_attentive_normalization, padding, normalization, dropblock_scheduler = unpack_keys(params_dict)
        ShortcutPooling = operations_dictionary['ShortcutPooling']
        self.normalization = normalization
        if dropblock_params == None:
            self.dropblock_params = None
        else:
            self.dropblock_params = deepcopy(dropblock_params)
            self.dropblock_scheduler = deepcopy(dropblock_scheduler)
            self.current_prob = None
            if self.dropblock_scheduler[0] == True:
                self.amount_decrement = (1 - self.dropblock_params[0]) / self.dropblock_scheduler[1]
                self.current_prob = tf.Variable(1., trainable=False)
                self.final_prob = self.dropblock_params[0]
        self.attentive_normalization_K = attentive_normalization_K
        #initialize layers
        self.use_grouped_convolutions = deepcopy(use_grouped_convolutions)
        self.use_selective_kernel = deepcopy(use_selective_kernel)
        self.use_cbam = deepcopy(use_cbam)
        self.use_attentive_normalization = deepcopy(use_attentive_normalization)
        self.norm_list = []
        self.act_list = []
        self.conv_list = []
        #shortcut connection
        if self.in_filters != self.num_filters:
            if self.pool != None and self.pool[0] == True:
                self.shortcut_pool_operation = ShortcutPooling(pool_size=self.pool[1], padding=padding)
            self.shortcut_convolution_operation = instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True)
            self.shortcut_normalization_operation = instantiate_normalization_layer(self.num_filters, operations_dictionary, params_dict)
        #conv-norm-act
        if self.use_selective_kernel[0] == False:
            if self.use_grouped_convolutions[0] == False:
                #if using projection shortcut, use strided convolution
                if self.pool != None and self.pool[0] == True:
                    self.conv_list.append(instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict, strided_conv=True, pool=self.pool))
                else:
                    self.conv_list.append(instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict))
            else:
                self.num_groups, out_filters = calculate_group_convolution_parameters(self.use_grouped_convolutions, self.in_filters, self.num_filters)
                if self.use_grouped_convolutions[3] == True:
                    #if using projction shortcut, use strided convolution
                    if self.pool != None and self.pool[0] == True:
                        self.conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict, strided_conv=True, pool=self.pool, groups=self.num_groups))
                    else:
                        self.conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict, groups=self.num_groups))
                else:
                    temp_conv_list = []
                    for num_group in range(0, self.num_groups):
                        #if using projection shortcut, use strided convolution
                        if self.pool != None and self.pool[0] == True:
                            temp_conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict, strided_conv=True, pool=self.pool))
                        else:
                            temp_conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict))
                    self.conv_list.append(temp_conv_list)
            self.norm_list.append(instantiate_normalization_layer(self.num_filters, operations_dictionary, params_dict))
            self.act_list.append(instantiate_activation_layer(operations_dictionary, params_dict))
        else:
            self.conv_list.append(Selective_Kernel_Block(num_filters, operations_dictionary, params_dict, in_filters=self.in_filters, pool=self.pool))
        #conv-norm-add-act
        if self.use_grouped_convolutions[0] == False:
            self.conv_list.append(instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict))
        else:
            self.num_groups, out_filters = calculate_group_convolution_parameters(self.use_grouped_convolutions, self.in_filters, self.num_filters)
            if self.use_grouped_convolutions[3] == True:
                self.conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict, groups=self.num_groups))
            else:
                temp_conv_list = []
                for num_group in range(0, self.num_groups):
                    temp_conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict))
                self.conv_list.append(temp_conv_list)
        #optional attentive normalization block
        if self.use_attentive_normalization[0] == True:
            self.attentive_normalization_block = Attentive_Normalization_Block(self.num_filters, self.attentive_normalization_K, operations_dictionary, params_dict)
        else:
            self.norm_list.append(instantiate_normalization_layer(self.num_filters, operations_dictionary, params_dict, zero_init=True))
        self.act_list.append(instantiate_activation_layer(operations_dictionary, params_dict))
        #optional cbam block
        if self.use_cbam[0] == True:
            self.cbam_block = CBAM_Block(self.num_filters, operations_dictionary, params_dict)
        #addition layer for residual
        self.add_operation = tf.keras.layers.Add()
    #
    def call(self, x, training=True):
        shortcut = x
        #shortcut projection if exists
        if self.in_filters != self.num_filters:
            if self.pool != None and self.pool[0] == True:
                shortcut = self.shortcut_pool_operation(shortcut)
            shortcut = self.shortcut_convolution_operation(shortcut)
            shortcut = apply_normalization(shortcut, self.shortcut_normalization_operation, self.normalization, training)
        #dropblock
        if self.dropblock_params != None and training == True:
            shortcut = dropblock(shortcut, self.dropblock_params, self.params_dict, current_prob=self.current_prob)
        #conv-norm-act
        if self.use_selective_kernel[0] == False:
            if self.use_grouped_convolutions[0] == False:
                x = self.conv_list[0](x)
            else:
                if self.use_grouped_convolutions[3] == True:
                    x = self.conv_list[0](x)
                else:
                    x_in = tf.split(x, self.num_groups, axis=-1)
                    x = tf.concat([self.conv_list[0][num_group](x_group) for num_group, x_group in enumerate(x_in)], axis=-1)
            x = apply_normalization(x, self.norm_list[0], self.normalization, training)
            x = self.act_list[0](x)
        else:
            x = self.conv_list[0](x)
        #dropblock
        if self.dropblock_params != None and training == True:
            x = dropblock(x, self.dropblock_params, self.params_dict, current_prob=self.current_prob)
        #conv-norm
        if self.use_grouped_convolutions[0] == False:
            x = self.conv_list[1](x)
        else:
            if self.use_grouped_convolutions[3] == True:
                x = self.conv_list[1](x)
            else:
                x_in = tf.split(x, self.num_groups, axis=-1)
                x = tf.concat([self.conv_list[1][num_group](x_group) for num_group, x_group in enumerate(x_in)], axis=-1)
        if self.use_attentive_normalization[0] == True:
            x = self.attentive_normalization_block(x)
        else:
            x = apply_normalization(x, self.norm_list[1], self.normalization, training)
        #dropblock
        if self.dropblock_params != None and training == True:
            x = dropblock(x, self.dropblock_params, self.params_dict, current_prob=self.current_prob)
        #optional cbam
        if self.use_cbam[0] == True:
            x = self.cbam_block(x)
        #add residual to shortcut
        x = self.add_operation([x, shortcut])
        #activate
        x = self.act_list[1](x)
        return x
    #
    def get_config(self):
        return {'num_filters': self.num_filters,
                'dilation_rate': self.dilation_rate,
                'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict,
                'in_filters': self.in_filters,
                'pool': self.pool,
                'dropblock_params': self.dropblock_params,
                'attentive_normalization_K': self.attentive_normalization_K}

#bottleneck residual block (computes [[Conv1->Norm->Act->Conv3->Norm->Act->Conv1->Norm]+shortcut]->Act)
class Residual_Bottleneck_Block(tf.keras.Model):
    def __init__(self, num_filters, dilation_rate, operations_dictionary, params_dict, in_filters=None, pool=None, dropblock_params=None, dense_parameters=None, attentive_normalization_K=None, name='residual_bottleneck_block', **kwargs):
        super(Residual_Bottleneck_Block, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.num_filters = num_filters
        self.dilation_rate = dilation_rate
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        if in_filters == None:
            self.in_filters = num_filters 
        else:
            self.in_filters = in_filters
        if pool == None:
            self.pool = None
        else:
            self.pool = deepcopy(pool)
        #unpack relevant dictionary elements
        use_grouped_convolutions, use_selective_kernel, use_cbam, use_attentive_normalization, padding, normalization, dropblock_scheduler = unpack_keys(params_dict)
        ShortcutPooling = operations_dictionary['ShortcutPooling']
        self.normalization = normalization
        if dropblock_params == None:
            self.dropblock_params = None
        else:
            self.dropblock_params = deepcopy(dropblock_params)
            self.dropblock_scheduler = deepcopy(dropblock_scheduler)
            self.current_prob = None
            if self.dropblock_scheduler[0] == True:
                self.amount_decrement = (1 - self.dropblock_params[0]) / self.dropblock_scheduler[1]
                self.current_prob = tf.Variable(1., trainable=False)
                self.final_prob = self.dropblock_params[0]
        self.attentive_normalization_K = attentive_normalization_K
        #initialize layers
        self.use_grouped_convolutions = deepcopy(use_grouped_convolutions)
        self.use_selective_kernel = deepcopy(use_selective_kernel)
        self.use_cbam = deepcopy(use_cbam)
        self.use_attentive_normalization = deepcopy(use_attentive_normalization)
        self.norm_list = []
        self.act_list = []
        self.conv_list = []
        #number of filters in each part of bottleneck
        if self.use_grouped_convolutions[0] == True:
            self.num_filters_list = [num_filters, num_filters, 2*num_filters]
        else:
            self.num_filters_list = [num_filters, num_filters, 4*num_filters]
        #shortcut connection
        if self.in_filters != self.num_filters_list[2]:
            if self.pool != None and self.pool[0] == True:
                self.shortcut_pool_operation = ShortcutPooling(pool_size=self.pool[1], padding=padding)
            self.shortcut_convolution_operation = instantiate_convolution_layer(self.num_filters_list[2], dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True)
            self.shortcut_normalization_operation = instantiate_normalization_layer(self.num_filters_list[2], operations_dictionary, params_dict)
        #first convolution is bottleneck
        self.conv_list.append(instantiate_convolution_layer(self.num_filters_list[0], dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True))
        self.norm_list.append(instantiate_normalization_layer(self.num_filters_list[0], operations_dictionary, params_dict))
        self.act_list.append(instantiate_activation_layer(operations_dictionary, params_dict))
        #second convolution is either standard, grouped, or selective kernel style
        if self.use_selective_kernel[0] == False:
            if self.use_grouped_convolutions[0] == False:
                #if using projection shortcut, use strided convolution
                if self.pool != None and self.pool[0] == True:
                    self.conv_list.append(instantiate_convolution_layer(self.num_filters_list[1], dilation_rate, operations_dictionary, params_dict, strided_conv=True, pool=self.pool))
                else:
                    self.conv_list.append(instantiate_convolution_layer(self.num_filters_list[1], dilation_rate, operations_dictionary, params_dict))
                if self.use_attentive_normalization[0] == False:
                    self.norm_list.append(instantiate_normalization_layer(self.num_filters_list[1], operations_dictionary, params_dict))
            else:
                self.num_groups, out_filters = calculate_group_convolution_parameters(self.use_grouped_convolutions, self.num_filters_list[0], self.num_filters_list[1])
                if self.use_grouped_convolutions[3] == True:
                    #if using projection shortcut, use strided convolution
                    if self.pool != None and self.pool[0] == True:
                        self.conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict, strided_conv=True, pool=self.pool, groups=self.num_groups))
                    else:
                        self.conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict, groups=self.num_groups))
                else:
                    temp_conv_list = []
                    for num_group in range(0, self.num_groups):
                        #if using projection shortcut, use strided convolution
                        if self.pool != None and self.pool[0] == True:
                            temp_conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict, strided_conv=True, pool=self.pool))
                        else:
                            temp_conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict))
                    self.conv_list.append(temp_conv_list)
                if self.use_attentive_normalization[0] == False:
                    self.norm_list.append(instantiate_normalization_layer(self.num_filters_list[1], operations_dictionary, params_dict, num_groups=self.num_groups))
            #optional attentive normalization block
            if self.use_attentive_normalization[0] == True:
                self.attentive_normalization_block = Attentive_Normalization_Block(self.num_filters_list[1], self.attentive_normalization_K, operations_dictionary, params_dict)
            self.act_list.append(instantiate_activation_layer(operations_dictionary, params_dict))
        else:
            self.conv_list.append(Selective_Kernel_Block(num_filters, operations_dictionary, params_dict, in_filters=self.num_filters_list[0], pool=self.pool))
        #final convolution is bottleneck back to original feature map size to allow for addition
        self.conv_list.append(instantiate_convolution_layer(self.num_filters_list[2], dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True))
        self.norm_list.append(instantiate_normalization_layer(self.num_filters_list[2], operations_dictionary, params_dict, zero_init=True))
        #optional cbam block
        if self.use_cbam[0] == True:
            self.cbam_block = CBAM_Block(self.num_filters_list[2], operations_dictionary, params_dict)
        #addition layer for residual
        self.add_operation = tf.keras.layers.Add()
        self.act_list.append(instantiate_activation_layer(operations_dictionary, params_dict))
    #
    def call(self, x, training=True):
        shortcut = x
        #shortcut projection if exists
        if self.in_filters != self.num_filters_list[2]:
            if self.pool != None and self.pool[0] == True:
                shortcut = self.shortcut_pool_operation(shortcut)
            shortcut = self.shortcut_convolution_operation(shortcut)
            shortcut = apply_normalization(shortcut, self.shortcut_normalization_operation, self.normalization, training)
        #dropblock
        if self.dropblock_params != None and training == True:
            shortcut = dropblock(shortcut, self.dropblock_params, self.params_dict, current_prob=self.current_prob)
        #first conv
        x = self.conv_list[0](x)
        x = apply_normalization(x, self.norm_list[0], self.normalization, training)
        x = self.act_list[0](x)
        #dropblock
        if self.dropblock_params != None and training == True:
            x = dropblock(x, self.dropblock_params, self.params_dict, current_prob=self.current_prob)
        #second conv
        if self.use_selective_kernel[0] == False:
            if self.use_grouped_convolutions[0] == False:
                x = self.conv_list[1](x)
            else:
                if self.use_grouped_convolutions[3] == True:
                    x = self.conv_list[1](x)
                else:
                    x_in = tf.split(x, self.num_groups, axis=-1)
                    x = tf.concat([self.conv_list[1][num_group](x_group) for num_group, x_group in enumerate(x_in)], axis=-1)
            if self.use_attentive_normalization[0] == True:
                x = self.attentive_normalization_block(x)
            else:
                x = apply_normalization(x, self.norm_list[1], self.normalization, training)
            x = self.act_list[1](x)
        else:
            x = self.conv_list[1](x)
        #dropblock
        if self.dropblock_params != None and training == True:
            x = dropblock(x, self.dropblock_params, self.params_dict, current_prob=self.current_prob)
        #third conv
        x = self.conv_list[2](x)
        x = apply_normalization(x, self.norm_list[-1], self.normalization, training)
        #dropblock
        if self.dropblock_params != None and training == True:
            x = dropblock(x, self.dropblock_params, self.params_dict, current_prob=self.current_prob)
        #optional cbam
        if self.use_cbam[0] == True:
            x = self.cbam_block(x)
        #add residual to shortcut
        x = self.add_operation([x, shortcut])
        #activate
        x = self.act_list[-1](x)
        return x
    #
    def get_config(self):
        return {'num_filters': self.num_filters,
                'dilation_rate': self.dilation_rate,
                'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict,
                'in_filters': self.in_filters,
                'pool': self.pool,
                'dropblock_params': self.dropblock_params,
                'attentive_normalization_K': self.attentive_normalization_K}

#standard dense block (computes [Conv3-Norm-Act]xN)
class Dense_Block(tf.keras.Model):
    def __init__(self, num_filters, dilation_rate, operations_dictionary, params_dict, in_filters=None, pool=None, dropblock_params=None, dense_parameters=None, attentive_normalization_K=None, name='dense_block', **kwargs):
        super(Dense_Block, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.num_filters = num_filters
        self.dilation_rate = dilation_rate
        self.dense_parameters = dense_parameters
        k, N1, N2 = self.dense_parameters
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        if in_filters == None:
            self.in_filters = num_filters
        else:
            self.in_filters = in_filters
        if pool == None:
            self.pool = None
        else:
            self.pool = deepcopy(pool)
        #unpack relevant dictionary elements
        use_grouped_convolutions, use_selective_kernel, use_cbam, use_attentive_normalization, preactivation, normalization, dense_bottleneck, dropblock_scheduler = unpack_keys(params_dict)
        self.preactivation = preactivation
        self.normalization = normalization
        self.dense_bottleneck = dense_bottleneck
        if dropblock_params == None:
            self.dropblock_params = None
        else:
            self.dropblock_params = deepcopy(dropblock_params)
            self.dropblock_scheduler = deepcopy(dropblock_scheduler)
            self.current_prob = None
            if self.dropblock_scheduler[0] == True:
                self.amount_decrement = (1 - self.dropblock_params[0]) / self.dropblock_scheduler[1]
                self.current_prob = tf.Variable(1., trainable=False)
                self.final_prob = self.dropblock_params[0]
        self.attentive_normalization_K = attentive_normalization_K
        #initialize layers
        self.use_grouped_convolutions = deepcopy(use_grouped_convolutions)
        self.use_grouped_convolutions[0] = False
        self.use_selective_kernel = deepcopy(use_selective_kernel)
        self.use_cbam = deepcopy(use_cbam)
        self.use_attentive_normalization = deepcopy(use_attentive_normalization)
        self.norm_list = []
        self.act_list = []
        self.conv_list = []
        if self.dense_bottleneck[0] == True:
            self.norm_list_bottleneck = []
            self.act_list_bottleneck = []
            self.conv_list_bottleneck = []
        # 1x1x1 transition layer conv
        if self.dense_parameters[1] == 0:
            self.transition_conv = instantiate_convolution_layer(self.num_filters // 2, dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True)
            self.transition_norm = instantiate_normalization_layer(self.num_filters // 2, operations_dictionary, params_dict)
            self.transition_act = instantiate_activation_layer(operations_dictionary, params_dict)
        #number of filters in each section of the block
        self.num_filters_list = [self.num_filters // 2 + k * i for i in range(N1, N2 + 1)]
        for i in range(0, N2 - N1):
            #optional bottleneck layers
            if self.dense_bottleneck[0] == True:
                if self.dense_bottleneck[2] == False or self.num_filters_list[i] >= k * self.dense_bottleneck[1]:
                    self.conv_list_bottleneck.append(instantiate_convolution_layer(k * self.dense_bottleneck[1], dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True))
                    self.norm_list_bottleneck.append(instantiate_normalization_layer(k * self.dense_bottleneck[1], operations_dictionary, params_dict))
                    self.act_list_bottleneck.append(instantiate_activation_layer(operations_dictionary, params_dict))
                else:
                    self.conv_list_bottleneck.append(None)
                    self.norm_list_bottleneck.append(None)
                    self.act_list_bottleneck.append(None)
            #convolution with 'k' filters
            if self.use_grouped_convolutions[0] == False:
                self.conv_list.append(instantiate_convolution_layer(k, dilation_rate, operations_dictionary, params_dict))
                self.norm_list.append(instantiate_normalization_layer(k, operations_dictionary, params_dict))
            else:
                if self.dense_bottleneck[0] == True and (self.dense_bottleneck[2] == False or self.num_filters_list[i] >= k * self.dense_bottleneck[1]):
                    self.num_groups, out_filters = calculate_group_convolution_parameters(self.use_grouped_convolutions, k * self.dense_bottleneck[1], k)
                else:
                    self.num_groups, out_filters = calculate_group_convolution_parameters(self.use_grouped_convolutions, self.num_filters_list[i], k)
                if self.use_grouped_convolutions[3] == True:
                    self.conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict, groups=self.num_groups))
                else:
                    temp_conv_list = []
                    for num_group in range(0, self.num_groups):
                        temp_conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict))
                    self.conv_list.append(temp_conv_list)
                self.norm_list.append(instantiate_normalization_layer(k, operations_dictionary, params_dict, num_groups=self.num_groups))
            self.act_list.append(instantiate_activation_layer(operations_dictionary, params_dict))
        #extra norm/act/conv in case using special blocks
        if self.use_cbam[0] == True:
            self.cbam_block = CBAM_Block(self.num_filters_list[-1], operations_dictionary, params_dict)
            self.final_norm = instantiate_normalization_layer(self.num_filters_list[-1], operations_dictionary, params_dict)
            self.final_act = instantiate_activation_layer(operations_dictionary, params_dict)
            self.final_conv = instantiate_convolution_layer(self.num_filters_list[-1], dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True)
        if self.use_selective_kernel[0] == True:
            self.selective_kernel_block = Selective_Kernel_Block(self.num_filters_list[-1], operations_dictionary, params_dict)
            if self.preactivation[0] == True:
                self.final_norm = instantiate_normalization_layer(self.num_filters_list[-1], operations_dictionary, params_dict)
                self.final_act = instantiate_activation_layer(operations_dictionary, params_dict)
                self.final_conv = instantiate_convolution_layer(self.num_filters_list[-1], dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True)
        if self.use_attentive_normalization[0] == True:
            self.attentive_normalization_block = Attentive_Normalization_Block(self.num_filters_list[-1], self.attentive_normalization_K, operations_dictionary, params_dict)
            self.final_act = instantiate_activation_layer(operations_dictionary, params_dict)
            self.final_conv = instantiate_convolution_layer(self.num_filters_list[-1], dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True)

    #
    def call(self, x, training=True):
        #transition layer
        layer_shape = x.get_shape().as_list()
        if layer_shape[-1] != self.num_filters // 2 and self.dense_parameters[1] == 0:
            if self.preactivation[0] == False:
                x = self.transition_conv(x)
            x = apply_normalization(x, self.transition_norm, self.normalization, training)
            x = self.transition_act(x)
            #dropblock
            if self.dropblock_params != None and training == True:
                x = dropblock(x, self.dropblock_params, self.params_dict, current_prob=self.current_prob)
            if self.preactivation[0] == True:
                x = self.transition_conv(x)
        #dense block
        for i in range(0, len(self.conv_list)):
            x_input = x
            #optional bottleneck layers
            if self.dense_bottleneck[0] == True and (self.dense_bottleneck[2] == False or self.num_filters_list[i] >= self.dense_parameters[0] * self.dense_bottleneck[1]):
                if self.preactivation[0] == False:
                    x = self.conv_list_bottleneck[i](x)
                x = apply_normalization(x, self.norm_list_bottleneck[i], self.normalization, training)
                x = self.act_list_bottleneck[i](x)
                if self.preactivation[0] == True:
                    x = self.conv_list_bottleneck[i](x)
            #standard convolution with K filters
            if self.preactivation[0] == True:
                x = apply_normalization(x, self.norm_list[i], self.normalization, training)
                x = self.act_list[i](x)
                #dropblock
                if self.dropblock_params != None and training == True:
                    x = dropblock(x, self.dropblock_params, self.params_dict, current_prob=self.current_prob)
            if self.use_grouped_convolutions[0] == False:
                x = self.conv_list[i](x)
            else:
                if self.use_grouped_convolutions[3] == True:
                    x = self.conv_list[i](x)
                else:
                    x_in = tf.split(x, self.num_groups, axis=-1)
                    x = tf.concat([self.conv_list[i][num_group](x_group) for num_group, x_group in enumerate(x_in)], axis=-1)
            if self.preactivation[0] == False:
                x = apply_normalization(x, self.norm_list[i], self.normalization, training)
                x = self.act_list[i](x)
                #dropblock
                if self.dropblock_params != None and training == True:
                    x = dropblock(x, self.dropblock_params, self.params_dict, current_prob=self.current_prob)
            #concatentation
            x = tf.concat([x, x_input], axis=-1)
        #optional special blocks
        if self.use_cbam[0] == True:
            if self.preactivation[0] == False:
                x = self.final_conv(x)
            x = apply_normalization(x, self.final_norm, self.normalization, training)
            x = self.cbam_block(x)
            x = self.final_act(x)
            if self.preactivation[0] == True:
                x = self.final_conv(x)
        if self.use_selective_kernel[0] == True:
            if self.preactivation[0] == True:
                x = apply_normalization(x, self.final_norm, self.normalization, training)
                x = self.final_act(x)
            x = self.selective_kernel_block(x)
            if self.preactivation[0] == True:
                x = self.final_conv(x)
        if self.use_attentive_normalization[0] == True:
            if self.preactivation[0] == False:
                x = self.final_conv(x)
            x = self.attentive_normalization_block(x)
            x = self.final_act(x)
            if self.preactivation[0] == True:
                x = self.final_conv(x)
        return x
    #
    def get_config(self):
        return {'num_filters': self.num_filters,
                'dilation_rate': self.dilation_rate,
                'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict,
                'in_filters': self.in_filters,
                'pool': self.pool,
                'dropblock_params': self.dropblock_params,
                'dense_parameters': self.dense_parameters,
                'attentive_normalization_K': self.attentive_normalization_K}

#standard convolutional block (computes Conv3-Norm-Act)
class Convolutional_Block(tf.keras.Model):
    def __init__(self, num_filters, dilation_rate, operations_dictionary, params_dict, in_filters=None, pool=None, dropblock_params=None, dense_parameters=None, attentive_normalization_K=None, name='convolutional_block', **kwargs):
        super(Convolutional_Block, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.num_filters = num_filters
        self.dilation_rate = dilation_rate
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        if in_filters == None:
            self.in_filters = num_filters
        else:
            self.in_filters = in_filters
        if pool == None:
            self.pool = None
        else:
            self.pool = deepcopy(pool)
        #unpack relevant dictionary elements
        use_grouped_convolutions, use_selective_kernel, use_cbam, use_attentive_normalization, preactivation, normalization, dropblock_scheduler = unpack_keys(params_dict)
        self.preactivation = preactivation
        self.normalization = normalization
        if dropblock_params == None:
            self.dropblock_params = None
        else:
            self.dropblock_params = deepcopy(dropblock_params)
            self.dropblock_scheduler = deepcopy(dropblock_scheduler)
            self.current_prob = None
            if self.dropblock_scheduler[0] == True:
                self.amount_decrement = (1 - self.dropblock_params[0]) / self.dropblock_scheduler[1]
                self.current_prob = tf.Variable(1., trainable=False)
                self.final_prob = self.dropblock_params[0]
        self.attentive_normalization_K = attentive_normalization_K
        #initialize layers
        self.use_grouped_convolutions = deepcopy(use_grouped_convolutions)
        self.use_selective_kernel = deepcopy(use_selective_kernel)
        self.use_cbam = deepcopy(use_cbam)
        self.use_attentive_normalization = deepcopy(use_attentive_normalization)
        if self.use_selective_kernel[0] == False:
            if self.use_grouped_convolutions[0] == False:
                self.convolution_operation = instantiate_convolution_layer(num_filters, dilation_rate, operations_dictionary, params_dict)
            else:
                self.num_groups, out_filters = calculate_group_convolution_parameters(self.use_grouped_convolutions, self.in_filters, self.num_filters)
                if self.use_grouped_convolutions[3] == True:
                    self.convolution_operation = instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict, groups=self.num_groups)
                else:
                    self.conv_list = []
                    for num_group in range(0, self.num_groups):
                        self.conv_list.append(instantiate_convolution_layer(out_filters, dilation_rate, operations_dictionary, params_dict))
            if self.use_attentive_normalization[0] == False:
                self.normalization_operation = instantiate_normalization_layer(num_filters, operations_dictionary, params_dict)
            else:
                #optional attentive normalization block
                self.attentive_normalization_block = Attentive_Normalization_Block(self.num_filters, self.attentive_normalization_K, operations_dictionary, params_dict)
            if self.use_cbam[0] == True:
                self.cbam_block = CBAM_Block(num_filters, operations_dictionary, params_dict)
            self.activation_operation = instantiate_activation_layer(operations_dictionary, params_dict)
        else:
            self.sk_block = Selective_Kernel_Block(num_filters, operations_dictionary, params_dict, in_filters=self.in_filters)
            if self.preactivation[0] == True:
                self.final_norm = instantiate_normalization_layer(self.num_filters, operations_dictionary, params_dict)
                self.final_act = instantiate_activation_layer(operations_dictionary, params_dict)
                self.final_conv = instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True)
    #
    def call(self, x, training=True):
        if self.use_selective_kernel[0] == False:
            if self.preactivation[0] == True:
                if self.use_attentive_normalization[0] == False:
                    x = apply_normalization(x, self.normalization_operation, self.normalization, training)
                else:
                    x = self.attentive_normalization_block(x)
                if self.use_cbam[0] == True:
                    x = self.cbam_block(x)
                x = self.activation_operation(x)
                #dropblock
                if self.dropblock_params != None and training == True:
                    x = dropblock(x, self.dropblock_params, self.operations_dictionary, self.params_dict, current_prob=self.current_prob)
            if self.use_grouped_convolutions[0] == False:
                x = self.convolution_operation(x)
            else:
                if self.use_grouped_convolutions[3] == True:
                    x = self.convolution_operation(x)
                else:
                    x_in = tf.split(x, self.num_groups, axis=-1)
                    x = tf.concat([self.conv_list[num_group](x_group) for num_group, x_group in enumerate(x_in)], axis=-1)
            if self.preactivation[0] == False:
                if self.use_attentive_normalization[0] == False:
                    x = apply_normalization(x, self.normalization_operation, self.normalization, training)
                else:
                    x = self.attentive_normalization_block(x)
                if self.use_cbam[0] == True:
                    x = self.cbam_block(x)
                x = self.activation_operation(x)
                #dropblock
                if self.dropblock_params != None and training == True:
                    x = dropblock(x, self.dropblock_params, self.operations_dictionary, self.params_dict, current_prob=self.current_prob)
        else:
            if self.preactivation[0] == True:
                x = self.final_norm(x)
                x = self.final_act(x)
                #dropblock
                if self.dropblock_params != None and training == True:
                    x = dropblock(x, self.dropblock_params, self.operations_dictionary, self.params_dict, current_prob=self.current_prob)
            x = self.sk_block(x)
            if self.preactivation[0] == True:
                x = self.final_conv(x)
            else:
                #dropblock
                if self.dropblock_params != None and training == True:
                    x = dropblock(x, self.dropblock_params, self.operations_dictionary, self.params_dict, current_prob=self.current_prob)
        return x
    #
    def get_config(self):
        return {'num_filters': self.num_filters,
                'dilation_rate': self.dilation_rate,
                'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict,
                'in_filters': self.in_filters,
                'pool': self.pool,
                'dropblock_params': self.dropblock_params,
                'attentive_normalization_K': self.attentive_normalization_K}

#selective-kernel block
class Selective_Kernel_Block(tf.keras.Model):
    def __init__(self, num_filters, operations_dictionary, params_dict, in_filters=None, pool=None, name='selective_kernel_block', **kwargs):
        super(Selective_Kernel_Block, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.num_filters = num_filters
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        if in_filters == None:
            self.in_filters = num_filters
        else:
            self.in_filters = in_filters
        if pool == None:
            self.pool = None
        else:
            self.pool = deepcopy(pool)
        #unpack relevant dictionary elements
        use_selective_kernel, use_grouped_convolutions, normalization = unpack_keys(params_dict)
        GlobalAveragePoolOperation, GlobalMaxPoolOperation = unpack_keys(operations_dictionary)
        self.normalization = normalization
        #bottleneck dimension
        self.use_selective_kernel = deepcopy(use_selective_kernel)
        self.use_grouped_convolutions = deepcopy(use_grouped_convolutions)
        bottleneck_dim = int(max(self.in_filters // self.use_selective_kernel[2], self.use_selective_kernel[3]))
        #initialize layers for SK block
        self.conv_list = []
        for num_branch in range(1, self.use_selective_kernel[1]+1):
            if self.use_grouped_convolutions[0] == False:
                #if using projection shortcut, use strided convolution
                if self.pool != None and self.pool[0] == True:
                    self.conv_list.append(instantiate_convolution_layer(num_filters, num_branch, operations_dictionary, params_dict, strided_conv=True, pool=self.pool))
                else:
                    self.conv_list.append(instantiate_convolution_layer(num_filters, num_branch, operations_dictionary, params_dict))
            else:
                self.num_groups, out_filters = calculate_group_convolution_parameters(self.use_grouped_convolutions, self.in_filters, self.num_filters)
                if self.use_grouped_convolutions[3] == True:
                    #if using projection shortcut, use strided convolution
                    if self.pool != None and self.pool[0] == True:
                        self.conv_list.append(instantiate_convolution_layer(out_filters, num_branch, operations_dictionary, params_dict, strided_conv=True, pool=self.pool, groups=self.num_groups))
                    else:
                        self.conv_list.append(instantiate_convolution_layer(out_filters, num_branch, operations_dictionary, params_dict, groups=self.num_groups))
                else:
                    conv_list_temp = []
                    for num_group in range(0, self.num_groups):
                        #if using projection shortcut, use strided convolution
                        if self.pool != None and self.pool[0] == True:
                            conv_list_temp.append(instantiate_convolution_layer(out_filters, num_branch, operations_dictionary, params_dict, strided_conv=True, pool=self.pool))
                        else:
                            conv_list_temp.append(instantiate_convolution_layer(out_filters, num_branch, operations_dictionary, params_dict))
                    self.conv_list.append(conv_list_temp)
        #norm and activation layers
        self.norm_list = []
        self.act_list = []
        for num_branch in range(0, self.use_selective_kernel[1]):
            if self.use_grouped_convolutions[0] == False:
                self.norm_list.append(instantiate_normalization_layer(num_filters, operations_dictionary, params_dict))
            else:
                self.norm_list.append(instantiate_normalization_layer(num_filters, operations_dictionary, params_dict, num_groups=self.num_groups))
            self.act_list.append(instantiate_activation_layer(operations_dictionary, params_dict))
        #fusing and reduction
        self.add_operation1 = tf.keras.layers.Add()
        self.global_average_pool_operation = GlobalAveragePoolOperation() 
        self.global_max_pool_operation = GlobalMaxPoolOperation()
        #shared MLP
        self.dense_operation1 = instantiate_dense_layer(bottleneck_dim, 'relu', operations_dictionary, params_dict)
        self.dense_operation2 = instantiate_dense_layer(int(self.use_selective_kernel[1]*num_filters), None, operations_dictionary, params_dict)
        #combine
        self.add_operation2 = tf.keras.layers.Add()
        self.softmax_operation = tf.keras.layers.Activation('softmax')
        self.multiply_list = []
        for num_branch in range(0, self.use_selective_kernel[1]):
            self.multiply_list.append(tf.keras.layers.Multiply())
        self.add_operation3 = tf.keras.layers.Add()
        #spatial selection
        self.spatial_conv = instantiate_convolution_layer(1, 1, operations_dictionary, params_dict, custom_kernel_size=7, activation='sigmoid', weight_standardization_override=True)
        self.spatial_multiply_operation = tf.keras.layers.Multiply()
    #
    def call(self, x, training=True):
        #convolutions with different dilations
        x_list = []
        if self.use_grouped_convolutions[0] == True:
            x_in = tf.split(x, self.num_groups, axis=-1)
        for num_branch in range(0, self.use_selective_kernel[1]):
            if self.use_grouped_convolutions[0] == True:
                if self.use_grouped_convolutions[3] == True:
                    x_list.append(self.conv_list[num_branch](x))
                else:
                    x_list.append(tf.concat([self.conv_list[num_branch][num_group](x_group) for num_group, x_group in enumerate(x_in)], axis=-1))
            else:
                x_list.append(self.conv_list[num_branch](x))
        #normalization and activation of both branches
        x_list_norm_act = []
        for num_branch in range(0, self.use_selective_kernel[1]):
            x_list_norm_act.append(self.act_list[num_branch](apply_normalization(x_list[num_branch], self.norm_list[num_branch], self.normalization, training)))
        #fuse
        x_fuse = self.add_operation1(x_list_norm_act)
        x_fuse_average = self.global_average_pool_operation(x_fuse)
        x_fuse_max = self.global_max_pool_operation(x_fuse)
        #average pool
        x_fuse_average = self.dense_operation1(x_fuse_average)
        x_fuse_average = self.dense_operation2(x_fuse_average)
        #max pool
        x_fuse_max = self.dense_operation1(x_fuse_max)
        x_fuse_max = self.dense_operation2(x_fuse_max)
        #sum and softmax
        x_sum = self.add_operation2([x_fuse_average, x_fuse_max])
        x_mixing = tf.stack(tf.split(x_sum, num_or_size_splits=self.use_selective_kernel[1], axis=-1), axis=-1)
        x_softmax = self.softmax_operation(x_mixing)
        #selection
        x_channel_select_list = []
        for num_branch in range(0, self.use_selective_kernel[1]):
            x_channel_select_list.append(self.multiply_list[num_branch]([x_list_norm_act[num_branch], x_softmax[...,num_branch]]))
        x_out_channel_select = self.add_operation3(x_channel_select_list)
        #now apply spatial activation to channel selected output
        x_spatial_average = tf.reduce_mean(x_out_channel_select, keepdims=True, axis=-1)
        x_spatial_max = tf.reduce_max(x_out_channel_select, keepdims=True, axis=-1)
        x_concat = tf.concat([x_spatial_average, x_spatial_max], axis=-1)
        x_spatial_conv = self.spatial_conv(x_concat)
        x_out_spatial_select = self.spatial_multiply_operation([x_out_channel_select, x_spatial_conv])
        return x_out_spatial_select
    #
    def get_config(self):
        return {'num_filters': self.num_filters,
                'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict,
                'in_filters': self.in_filters,
                'pool': self.pool}

#convolutional block attention module
class CBAM_Block(tf.keras.Model):
    def __init__(self, num_filters, operations_dictionary, params_dict, name='cbam_block', **kwargs):
        super(CBAM_Block, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.num_filters = num_filters
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        #unpack relevant dictionary elements
        use_cbam = params_dict['use_cbam']
        GlobalAveragePoolOperation, GlobalMaxPoolOperation = unpack_keys(operations_dictionary)
        #bottleneck dimension
        self.use_cbam = deepcopy(use_cbam)
        bottleneck_dim = int(max(self.num_filters // self.use_cbam[1], self.use_cbam[2]))
        #initialize layers for channel activation
        #reduction
        self.global_average_pool_operation = GlobalAveragePoolOperation() 
        self.global_max_pool_operation = GlobalMaxPoolOperation()
        #shared MLP
        self.dense_operation1 = instantiate_dense_layer(bottleneck_dim, 'relu', operations_dictionary, params_dict)
        self.dense_operation2 = instantiate_dense_layer(num_filters, None, operations_dictionary, params_dict)
        #combine
        self.add_operation = tf.keras.layers.Add()
        self.sigmoid_operation = tf.keras.layers.Activation('sigmoid')
        self.channel_multiply_operation = tf.keras.layers.Multiply()
        #initialize layers for spatial activation 
        self.spatial_conv = instantiate_convolution_layer(1, 1, operations_dictionary, params_dict, custom_kernel_size=7, activation='sigmoid', weight_standardization_override=True)
        self.spatial_multiply_operation = tf.keras.layers.Multiply()
    #
    def call(self, x, training=True):
        #channel activation
        #global pooling
        x_average = self.global_average_pool_operation(x)
        x_max = self.global_max_pool_operation(x)
        #average pool through shared MLP
        x_average = self.dense_operation1(x_average)
        x_average = self.dense_operation2(x_average)
        #max pool through shared MLP
        x_max = self.dense_operation1(x_max)
        x_max = self.dense_operation2(x_max)
        #sum and sigmoid
        x_sum = self.add_operation([x_average, x_max])
        x_sigmoid = self.sigmoid_operation(x_sum)
        x_out_channel = self.channel_multiply_operation([x, x_sigmoid])
        #spatial activation
        x_out_spatial_average = tf.reduce_mean(x_out_channel, keepdims=True, axis=-1)
        x_out_spatial_max = tf.reduce_max(x_out_channel, keepdims=True, axis=-1)
        x_concat = tf.concat([x_out_spatial_average, x_out_spatial_max], axis=-1)
        x_out_spatial_conv = self.spatial_conv(x_concat)
        x_out_spatial = self.spatial_multiply_operation([x_out_channel, x_out_spatial_conv])
        return x_out_spatial
    #
    def get_config(self):
        return {'num_filters': self.num_filters,
                'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict}

#attentive normalization
class Attentive_Normalization_Block(tf.keras.Model):
    def __init__(self, num_filters, attentive_normalization_K, operations_dictionary, params_dict, name='attentive_normalization_block', **kwargs):
        super(Attentive_Normalization_Block, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.num_filters = num_filters
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        #unpack relevant dictionary elements
        use_attentive_normalization, normalization, mixed_precision = unpack_keys(params_dict)
        GlobalAveragePoolOperation, GlobalMaxPoolOperation = unpack_keys(operations_dictionary)
        self.normalization = normalization
        #bottleneck dimension
        self.use_attentive_normalization = deepcopy(use_attentive_normalization)
        bottleneck_dim = int(max(self.num_filters // self.use_attentive_normalization[1], self.use_attentive_normalization[2]))
        #initialize layers for learning K mixture variables (variable of size (batch_size, K))
        #reduction
        self.global_average_pool_operation = GlobalAveragePoolOperation() 
        self.global_max_pool_operation = GlobalMaxPoolOperation()
        #shared MLP
        self.dense_operation1 = instantiate_dense_layer(bottleneck_dim, 'relu', operations_dictionary, params_dict)
        self.dense_operation2 = instantiate_dense_layer(attentive_normalization_K, None, operations_dictionary, params_dict)
        #combine
        self.add_operation1 = tf.keras.layers.Add()
        self.hard_sigmoid_operation = tf.keras.layers.Activation('hard_sigmoid')
        #normalization
        self.normalization_operation = instantiate_normalization_layer(num_filters, operations_dictionary, params_dict, center=False)
        #K channel-wise affine transformation variables (variable of size (K, num_filters))
        if mixed_precision == True:
            self.gamma = tf.Variable(tf.random_normal_initializer(1., 0.1)(shape=[attentive_normalization_K, self.num_filters], dtype=tf.float16))
            self.beta = tf.Variable(tf.random_normal_initializer(0., 0.1)(shape=[attentive_normalization_K, self.num_filters], dtype=tf.float16))
        else:
            self.gamma = tf.Variable(tf.random_normal_initializer(1., 0.1)(shape=[attentive_normalization_K, self.num_filters]))
            self.beta = tf.Variable(tf.random_normal_initializer(0., 0.1)(shape=[attentive_normalization_K, self.num_filters]))
        self.multiply_operation = tf.keras.layers.Multiply()
        self.add_operation2 = tf.keras.layers.Add()
    #
    def call(self, x, training=True):
        #instance-specific attentive weights
        x_average = self.global_average_pool_operation(x)
        x_max = self.global_max_pool_operation(x)
        #average pool through shared MLP
        x_average = self.dense_operation1(x_average)
        x_average = self.dense_operation2(x_average)
        #max pool through shared MLP
        x_max = self.dense_operation1(x_max)
        x_max = self.dense_operation2(x_max)
        #sum and sigmoid
        x_sum = self.add_operation1([x_average, x_max])
        x_lambda = self.hard_sigmoid_operation(x_sum)
        #matrix multiply mixture variables with affine transformation variables (final variable of size (batch_size, num_filters))
        transformed_gamma = tf.matmul(x_lambda, self.gamma)
        transformed_beta = tf.matmul(x_lambda, self.beta)
        #normalization (without scaling or centering)
        x_standardized = apply_normalization(x, self.normalization_operation, self.normalization, training)
        #scale and center using transformed variables
        x_out = self.add_operation2([self.multiply_operation([transformed_gamma, x_standardized]), transformed_beta])
        return x_out
    #
    def get_config(self):
        return {'num_filters': self.num_filters,
                'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict}

#convolutional layer in encoding arm of network
class Encoding_Convolutional_Layer(tf.keras.Model):
    def __init__(self, num_filters, dilation_rate, pool, num_blocks, block_growth_rate, dropblock_params, attentive_normalization_K, operations_dictionary, params_dict, name='encoding_convolutional_layer', **kwargs):
        super(Encoding_Convolutional_Layer, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.num_filters = num_filters
        self.dilation_rate = dilation_rate
        self.pool = pool
        self.num_blocks = num_blocks
        self.dropblock_params = dropblock_params
        self.attentive_normalization_K = attentive_normalization_K
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        #unpack relevant dictionary elements
        block_type, use_grouped_convolutions, preactivation, num_inputs, pooling, normalization, anti_aliased_pooling = unpack_keys(params_dict)
        CustomBlock, PoolOperation = unpack_keys(operations_dictionary)
        self.preactivation = preactivation
        self.pooling = pooling
        self.normalization = normalization
        self.anti_aliased_pooling = anti_aliased_pooling
        #change num_filters amount to match incoming skip connection if using residual bottleneck
        if block_type == 'Residual_Bottleneck_Block':
            if use_grouped_convolutions[0] == True:
                self.num_filters = self.num_filters * 2
            else:
                self.num_filters = self.num_filters * 4
        #if using Dense Block, use proper number of filters so that we don't need to also use a transition layer
        if block_type == 'Dense_Block':
            self.num_filters = self.num_filters // 2
        #initialize layers
        if pool[0] == True:
            self.pool_operation_list = []
            if PoolOperation is not None:
                if self.preactivation[0] == True:
                    self.pool_operation_list.append(instantiate_normalization_layer(self.num_filters // 2, operations_dictionary, params_dict))
                    self.pool_operation_list.append(instantiate_activation_layer(operations_dictionary, params_dict))
                self.pool_operation_list.extend(instantiate_pooling_layer(pool, operations_dictionary, params_dict, num_filters=self.num_filters // 2))
                if self.preactivation[0] == True:
                    self.pool_operation_list.append(instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True))
            elif pooling == 'StridedConvolution':
                if self.preactivation[0] == False:
                    if self.anti_aliased_pooling[0] == False:
                        self.pool_operation_list.append(instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict, strided_conv=True, pool=pool))
                    else:
                        self.pool_operation_list.append(instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict))
                    self.pool_operation_list.append(instantiate_normalization_layer(self.num_filters, operations_dictionary, params_dict))
                else:
                    self.pool_operation_list.append(instantiate_normalization_layer(self.num_filters // 2, operations_dictionary, params_dict))
                self.pool_operation_list.append(instantiate_activation_layer(operations_dictionary, params_dict))
                if self.anti_aliased_pooling[0] == True:
                    self.pool_operation_list.extend(instantiate_pooling_layer(pool, operations_dictionary, params_dict, num_filters=self.num_filters))
                if self.preactivation[0] == True:
                    if self.anti_aliased_pooling[0] == False:
                        self.pool_operation_list.append(instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict, strided_conv=True, pool=pool))
                    else:
                        self.pool_operation_list.append(instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict))
                        self.pool_operation_list.extend(instantiate_pooling_layer(pool, operations_dictionary, params_dict, num_filters=self.num_filters))
        self.blocks_list_encoder = []
        total_number_dense_connections = num_filters // 2 // block_growth_rate
        dense_connections_per_block = np.linspace(0, total_number_dense_connections, self.num_blocks+1).astype(int)
        for i in range(0, num_blocks):
            #first layer in network
            if i == 0 and pool[0] == False and pool[2] == True:
                if self.preactivation[0] == True:
                    in_filters = min(num_filters, self.preactivation[1])
                else:
                    in_filters = num_inputs
            #first layer directly in level (either directly after pooling in normal case, or no pooling for atrous convolution)
            elif i == 0 and pooling != 'StridedConvolution' and (pool[0] == True or self.dilation_rate != 1) and self.preactivation[0] == False:
                in_filters = self.num_filters // 2
            else:
                in_filters = self.num_filters
            #parameters for dense block
            dense_parameters = [block_growth_rate, dense_connections_per_block[i], dense_connections_per_block[i+1]]
            #if applying pooling via shortcut, provide pool variable (only usable with residual block)
            if (self.pool[0] == True and len(self.pool_operation_list) > 0) or i > 0:
                self.blocks_list_encoder.append(CustomBlock(num_filters, dilation_rate, operations_dictionary, params_dict, in_filters=in_filters, dropblock_params=dropblock_params, dense_parameters=dense_parameters, attentive_normalization_K=attentive_normalization_K))
            else:
                self.blocks_list_encoder.append(CustomBlock(num_filters, dilation_rate, operations_dictionary, params_dict, in_filters=in_filters, pool=pool, dropblock_params=dropblock_params, dense_parameters=dense_parameters, attentive_normalization_K=attentive_normalization_K))
    #
    def call(self, x, training=True):
        if self.pool[0] == True and len(self.pool_operation_list) > 0:
            for i in range(0, len(self.pool_operation_list)):
                if self.preactivation[0] == True:
                    if i == 0:
                        #normalization of strided convolution or 1x1x1 conv
                        x = apply_normalization(x, self.pool_operation_list[i], self.normalization, training)
                    else:
                        x = self.pool_operation_list[i](x)
                else:
                    if self.pooling == 'StridedConvolution' and i == 1:
                        #normalization of strided convolution
                        x = apply_normalization(x, self.pool_operation_list[i], self.normalization, training)
                    else:
                        x = self.pool_operation_list[i](x)
        for i in range(0, len(self.blocks_list_encoder)):
            x = self.blocks_list_encoder[i](x)
        return x
    #
    def get_config(self):
        return {'num_filters': self.num_filters,
                'dilation_rate': self.dilation_rate,
                'pool': self.pool,
                'num_blocks': self.num_blocks,
                'dropblock_params': self.dropblock_params,
                'attentive_normalization_K': self.attentive_normalization_K,
                'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict}

#convolutional layer in decoding arm of network
class Decoding_Convolutional_Layer(tf.keras.Model):
    def __init__(self, num_filters, dilation_rate, kernel_size_transposed_conv, num_blocks, block_growth_rate, dropblock_params, attentive_normalization_K, operations_dictionary, params_dict, name='decoding_convolutional_layer', **kwargs):
        super(Decoding_Convolutional_Layer, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.num_filters = num_filters
        self.dilation_rate = dilation_rate
        self.kernel_size_transposed_conv = kernel_size_transposed_conv
        self.num_blocks = num_blocks
        self.dropblock_params = dropblock_params
        self.attentive_normalization_K = attentive_normalization_K
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        #unpack relevant dictionary elements
        block_type, use_grouped_convolutions, upsampling, preactivation, normalization, anti_aliased_pooling = unpack_keys(params_dict)
        CustomBlock = operations_dictionary['CustomBlock']
        self.preactivation = preactivation
        self.normalization = normalization
        #change num_filters amount to match incoming skip connection if using residual bottleneck
        if block_type == 'Residual_Bottleneck_Block':
            if use_grouped_convolutions[0] == True:
                self.num_filters = self.num_filters * 2
            else:
                self.num_filters = self.num_filters * 4
        #initialize layers
        self.upsampling = upsampling
        if self.upsampling[0] == 'Deconvolution':
            self.convolution_transpose_operation = instantiate_transposed_convolution_layer(self.num_filters, kernel_size_transposed_conv, operations_dictionary, params_dict)
        else:
            self.convolution_operation = instantiate_convolution_layer(self.num_filters, dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True, weight_standardization_override=True)
            self.upsampling_operation_list = instantiate_upsampling_layer(kernel_size_transposed_conv, self.num_filters, operations_dictionary, params_dict)
            if anti_aliased_pooling[1] == True and len(np.unique(kernel_size_transposed_conv)) == 1:
                pad_amount = [pad_val - 1 for pad_val in [4,4,4]]
            else:
                pad_amount = [pad_val - 1 for pad_val in kernel_size_transposed_conv]
            self.padding_values = [(0,0)] + [(pad_val // 2, pad_val - (pad_val // 2)) for pad_val in pad_amount] + [(0,0)]
        self.normalization_operation = instantiate_normalization_layer(self.num_filters, operations_dictionary, params_dict)
        self.activation_operation = instantiate_activation_layer(operations_dictionary, params_dict)
        self.blocks_list_decoder = []
        total_number_dense_connections = self.num_filters // 2 // block_growth_rate
        dense_connections_per_block = np.linspace(0, total_number_dense_connections, self.num_blocks+1).astype(int)
        for i in range(0, num_blocks):
            #first layer directly after concatenation layer
            if i == 0:
                in_filters = int(self.num_filters * 2)
            else:
                in_filters = self.num_filters
            #parameters for dense block
            dense_parameters = [block_growth_rate, dense_connections_per_block[i], dense_connections_per_block[i+1]]
            self.blocks_list_decoder.append(CustomBlock(num_filters, dilation_rate, operations_dictionary, params_dict, in_filters=in_filters, dropblock_params=dropblock_params, dense_parameters=dense_parameters, attentive_normalization_K=attentive_normalization_K))
    #
    def call(self, x, concat_layer, training=True):
        if self.preactivation[0] == True:
            x = apply_normalization(x, self.normalization_operation, self.normalization, training)
            x = self.activation_operation(x)
        if self.upsampling[0] == 'Deconvolution':
            x = self.convolution_transpose_operation(x)
        else:
            x = self.convolution_operation(x)
            x = self.upsampling_operation_list[0](x)
            #use symmetric padding to ensure that interpolation is correct
            #x = tf.pad(x, self.padding_values, "SYMMETRIC")
            x = tf.pad(x, self.padding_values)
            x = self.upsampling_operation_list[1](x)
        if self.preactivation[0] == False:
            x = apply_normalization(x, self.normalization_operation, self.normalization, training)
            x = self.activation_operation(x)
        x = tf.concat([x, concat_layer], axis=-1)
        for i in range(0, len(self.blocks_list_decoder)):
            x = self.blocks_list_decoder[i](x)
        return x
    #
    def get_config(self):
        return {'num_filters': self.num_filters,
                'dilation_rate': self.dilation_rate,
                'kernel_size_transposed_conv': self.kernel_size_transposed_conv,
                'num_blocks': self.num_blocks,
                'dropblock_params': self.dropblock_params,
                'attentive_normalization_K': self.attentive_normalization_K,
                'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict}

#Encoding arm of network
class Encoder(tf.keras.Model):
    def __init__(self, operations_dictionary, params_dict, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        #unpack relevant dictionary elements
        preactivation, levels, num_repeat_pooling, filter_size_conv_transpose_pool, atrous_conv, filter_num_per_level, blocks_per_encoder_level, dense_growth_rate_per_encoder_level, dropblock_params, use_attentive_normalization = unpack_keys(params_dict)
        self.preactivation = preactivation
        #initialize layers
        if self.preactivation[0] == True:
            #if using preactivation, first convolution uses kernel size 7
            self.convolution_operation = instantiate_convolution_layer(min(filter_num_per_level[0], self.preactivation[1]), 1, operations_dictionary, params_dict, custom_kernel_size=self.preactivation[2])
        self.layers_list = []
        for i in range(0, levels):
            if i < levels - atrous_conv[0]:
                dilation_rate = 1
                if i == 0:
                    pool = [False, filter_size_conv_transpose_pool, True]
                else:
                    #calculate pool kernel size for this level of the encoder
                    kernel_size_pool = [x if y == -1 or i <= y else 1 for x,y in zip(filter_size_conv_transpose_pool, num_repeat_pooling)]
                    pool = [True, kernel_size_pool]
            else:
                dilation_rate = tuple(atrous_conv[1] ** (-levels + i + atrous_conv[0] + 1))
                pool = [False, filter_size_conv_transpose_pool, False]
            self.layers_list.append(Encoding_Convolutional_Layer(filter_num_per_level[i], dilation_rate, pool, blocks_per_encoder_level[i], dense_growth_rate_per_encoder_level[i], dropblock_params[i], use_attentive_normalization[3][i], operations_dictionary, params_dict))
    #
    def call(self, x):
        if self.preactivation[0] == True:
            x = self.convolution_operation(x)
        skip_connection_list = []
        for i in range(0, len(self.layers_list)):
            x = self.layers_list[i](x)
            skip_connection_list.append(x)
        return x, skip_connection_list
    #
    def get_config(self):
        return {'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict}

#Decoding arm of network
class Decoder(tf.keras.Model):
    def __init__(self, operations_dictionary, params_dict, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        #unpack relevant dictionary elements
        preactivation, normalization, levels, num_repeat_pooling, filter_size_conv_transpose_pool, atrous_conv, filter_num_per_level, blocks_per_decoder_level, dense_growth_rate_per_decoder_level, deep_supervision, num_outputs, dropblock_params, anti_aliased_pooling, use_attentive_normalization, retina_net_bias_initializer, mixed_precision = unpack_keys(params_dict)
        self.preactivation = preactivation
        self.normalization = normalization
        #initialize layers
        self.mixed_precision = mixed_precision
        self.num_outputs = num_outputs
        self.deep_supervision_index = deepcopy(deep_supervision)
        filter_num_per_level = filter_num_per_level[::-1][1:]
        dropblock_params = dropblock_params[::-1][1:]
        self.layers_list_atrous = []
        self.layers_list_conv_transpose = []
        for i in range(0, levels-1):
            if i >= atrous_conv[0]:
                dilation_rate = 1
                #calculate transposed convolution kernel size for this level of the decoder
                kernel_size_transposed_conv = [x if y == -1 or i >= (levels - 1 - y) else 1 for x,y in zip(filter_size_conv_transpose_pool, num_repeat_pooling)]
                self.layers_list_conv_transpose.append(Decoding_Convolutional_Layer(filter_num_per_level[i], dilation_rate, kernel_size_transposed_conv, blocks_per_decoder_level[i], dense_growth_rate_per_decoder_level[i], dropblock_params[i], use_attentive_normalization[3][i], operations_dictionary, params_dict))
            else:
                pool = [False, filter_size_conv_transpose_pool, False]
                dilation_rate = tuple(atrous_conv[1] ** (atrous_conv[0] - i))
                self.layers_list_atrous.append(Encoding_Convolutional_Layer(filter_num_per_level[i], dilation_rate, pool, blocks_per_decoder_level[i], dense_growth_rate_per_decoder_level[i], dropblock_params[i], use_attentive_normalization[3][i], operations_dictionary, params_dict))
        #final convolution leading to output
        dilation_rate = 1
        self.convolution_operation = instantiate_convolution_layer(num_outputs, dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True, retina_net_bias_initializer=retina_net_bias_initializer, weight_standardization_override=True)
        if self.preactivation[0] == True:
            self.normalization_operation = instantiate_normalization_layer(filter_num_per_level[-1], operations_dictionary, params_dict)
            self.activation_operation = instantiate_activation_layer(operations_dictionary, params_dict)
        #optional deep supervision
        if deep_supervision[0] == True:
            self.deep_supervision_index.append(levels - 2 - deep_supervision[1])
            self.layers_list_seg = []
            self.layers_list_upsample = []
            self.layers_list_add = []
            self.padding_values_list = []
            if self.preactivation[0] == True:
                self.layers_list_norm = []
                self.layers_list_act = []
            for i in range(0, deep_supervision[1]):
                #calculate upsampling kernel size for this level of deep supervision
                kernel_size_upsampling = [x if y == -1 or (deep_supervision[1] - i) <= y else 1 for x,y in zip(filter_size_conv_transpose_pool, num_repeat_pooling)]
                self.layers_list_seg.append(instantiate_convolution_layer(num_outputs, dilation_rate, operations_dictionary, params_dict, segmentation_convolution=True, weight_standardization_override=True))
                self.layers_list_upsample.append(instantiate_upsampling_layer(kernel_size_upsampling, self.num_outputs, operations_dictionary, params_dict))
                self.layers_list_add.append(tf.keras.layers.Add())
                if anti_aliased_pooling[1] == True and len(np.unique(kernel_size_upsampling)) == 1:
                    pad_amount = [pad_val - 1 for pad_val in [4,4,4]]
                else:
                    pad_amount = [pad_val - 1 for pad_val in kernel_size_upsampling]
                self.padding_values_list.append([(0,0)] + [(pad_val // 2, pad_val - (pad_val // 2)) for pad_val in pad_amount] + [(0,0)])
                if self.preactivation[0] == True:
                    self.layers_list_norm.append(instantiate_normalization_layer(filter_num_per_level[self.deep_supervision_index[2] + i], operations_dictionary, params_dict))
                    self.layers_list_act.append(instantiate_activation_layer(operations_dictionary, params_dict))
        #optional final layer to cast back to float32 if using mixed_precision
        if self.mixed_precision == True:
            self.cast_to_float32_layer = tf.keras.layers.Activation('linear', dtype='float32')
    #
    def call(self, x, skip_connection_list, training=True):
        len_atrous_layers = len(self.layers_list_atrous)
        len_conv_transpose_layers = len(self.layers_list_conv_transpose)
        skip_connection_list = skip_connection_list[:(-1-len_atrous_layers)][::-1]
        deep_supervision_counter = 0
        for i in range(0, len_atrous_layers + len_conv_transpose_layers):
            if i < len(self.layers_list_atrous):
                x = self.layers_list_atrous[i](x)
            else:
                index_conv_transpose = i-len_atrous_layers
                x = self.layers_list_conv_transpose[index_conv_transpose](x, skip_connection_list[index_conv_transpose])
            if self.deep_supervision_index[0] == True and deep_supervision_counter < self.deep_supervision_index[1] and i >= self.deep_supervision_index[2]:
                x_deep_supervision = x
                if self.preactivation[0] == True:
                    x_deep_supervision = apply_normalization(x_deep_supervision, self.layers_list_norm[deep_supervision_counter], self.normalization, training)
                    x_deep_supervision = self.layers_list_act[deep_supervision_counter](x_deep_supervision)
                x_deep_supervision = self.layers_list_seg[deep_supervision_counter](x_deep_supervision)
                if deep_supervision_counter > 0:
                    x_deep_supervision = self.layers_list_add[deep_supervision_counter-1]([x_deep_supervision, x_deep_supervision_upsample])
                x_deep_supervision_upsample = self.layers_list_upsample[deep_supervision_counter][0](x_deep_supervision)
                #use symmetric padding to ensure that interpolation is correct
                #x_deep_supervision_upsample = tf.pad(x_deep_supervision_upsample, self.padding_values_list[deep_supervision_counter], "SYMMETRIC")
                x_deep_supervision_upsample = tf.pad(x_deep_supervision_upsample, self.padding_values_list[deep_supervision_counter])
                x_deep_supervision_upsample = self.layers_list_upsample[deep_supervision_counter][1](x_deep_supervision_upsample) 
                deep_supervision_counter = deep_supervision_counter + 1 
        if self.preactivation[0] == True:
            x = apply_normalization(x, self.normalization_operation, self.normalization, training)
            x = self.activation_operation(x)
        x = self.convolution_operation(x)
        if self.deep_supervision_index[0] == True:
            x = self.layers_list_add[deep_supervision_counter-1]([x, x_deep_supervision_upsample])
        if self.mixed_precision == True:
            x = self.cast_to_float32_layer(x)
        return x
    #
    def get_config(self):
        return {'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict}

#Full unet architecture combining encoding and decoding arms with "block" of choice
class Unet(tf.keras.Model):
    def __init__(self, operations_dictionary, params_dict, name='unet', **kwargs):
        super(Unet, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        #encoder/decoder arms
        self.encoder = Encoder(operations_dictionary, params_dict)
        self.decoder = Decoder(operations_dictionary, params_dict)
    #
    def call(self, inputs):
        #connect encoder and decoder
        x, skip_connection_list = self.encoder(inputs)
        outputs = self.decoder(x, skip_connection_list)
        return outputs
    #
    def get_config(self):
        return {'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict}

#Generic classification network architecture using "block" of choice (e.g. choosing residual block will make a ResNet)
class ClassificationNetwork(tf.keras.Model):
    def __init__(self, operations_dictionary, params_dict, name='net', **kwargs):
        super(ClassificationNetwork, self).__init__(name=name, **kwargs)
        #save input arguments for use in config method
        self.operations_dictionary = operations_dictionary
        self.params_dict = params_dict
        num_outputs, retina_net_bias_initializer, mixed_precision = unpack_keys(params_dict)
        self.mixed_precision = mixed_precision
        GlobalAveragePoolOperation = operations_dictionary['GlobalAveragePoolOperation']
        #encoder/decoder arms
        self.encoder = Encoder(operations_dictionary, params_dict)
        self.global_average_pool_operation = GlobalAveragePoolOperation()
        self.dense_operation = instantiate_dense_layer(num_outputs, 'linear', operations_dictionary, params_dict, retina_net_bias_initializer=retina_net_bias_initializer)
        if self.mixed_precision == True:
            self.cast_to_float32_layer = tf.keras.layers.Activation('linear', dtype='float32')
    #
    def call(self, inputs):
        #connect encoder and decoder
        x, _ = self.encoder(inputs)
        x = self.global_average_pool_operation(x)
        outputs = self.dense_operation(x)
        if self.mixed_precision == True:
            outputs = self.cast_to_float32_layer(outputs)
        return outputs
    #
    def get_config(self):
        return {'operations_dictionary': self.operations_dictionary,
                'params_dict': self.params_dict}

#Custom training step in order to facilitate weight decay of only convolutional filters 
class CustomModel(tf.keras.Model):
    def __init__(self, mixed_precision, custom_fit, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.mixed_precision = mixed_precision
        self.custom_fit = custom_fit
        #split weights into two groups (kernels vs. bias/scale)
        self.weights_kernel = [v for v in self.trainable_variables if 'normalization' not in v.name and 'bias' not in v.name]
        self.weights_kernel_indices = [i for i,v in enumerate(self.trainable_variables) if 'normalization' not in v.name and 'bias' not in v.name]
        self.weights_bias_and_norm = [v for v in self.trainable_variables if 'normalization' in v.name or 'bias' in v.name]
        self.weights_bias_and_norm_indices = [i for i,v in enumerate(self.trainable_variables) if 'normalization' in v.name or 'bias' in v.name]
    #
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)
            if self.mixed_precision == True:
                loss = self.optimizer.get_scaled_loss(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        if self.custom_fit[3] == True:
            #process gradients via gradient centralization (i.e. all gradients for conv/fc layers are centered to be zero mean)
            gradients = [tf.math.add(g, -tf.math.reduce_mean(g, axis=tuple(range(0,len(list(g.shape))-1)), keepdims = True)) if len(list(g.shape)) > 1 else g for g in gradients]
        if self.mixed_precision == True:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        if self.custom_fit[1] == True:
            #apply weight decay to all parameters
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        else:
            if self.custom_fit[2][0] == False:
                #apply decay only to kernel weights, leaving bias and scale terms un-regularized
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables), decay_var_list=self.weights_kernel)
            else:
                #additionally to not regularizing bias and scale terms, use different learning rate for these terms
                #start by applying update to kernel weights
                gradients_weights_kernel = [grad for i,grad in enumerate(gradients) if i in self.weights_kernel_indices]
                self.optimizer.apply_gradients(zip(gradients_weights_kernel, self.weights_kernel))
                #next, get bias and scale gradients
                gradients_bias_and_norm = [grad for i,grad in enumerate(gradients) if i in self.weights_bias_and_norm_indices]
                #preserve old learning rate and weight decay values
                current_lr = tf.keras.backend.get_value(self.optimizer.learning_rate)
                current_wd = tf.keras.backend.get_value(self.optimizer.weight_decay)
                #set a new learning rate to desired value; set weight decay to zero
                tf.keras.backend.set_value(self.optimizer.learning_rate, current_lr * self.custom_fit[2][1])
                tf.keras.backend.set_value(self.optimizer.weight_decay, 0.)
                #apply update to bias and scale
                self.optimizer.apply_gradients(zip(gradients_bias_and_norm, self.weights_bias_and_norm))
                #reset learning rate and weight decay for the next cycle
                tf.keras.backend.set_value(self.optimizer.learning_rate, current_lr)
                tf.keras.backend.set_value(self.optimizer.weight_decay, current_wd)
        #update metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    #
    def test_step(self, data):
        x, y = data
        # Compute predictions
        y_pred = self(x, training=False)
        # Updates the metrics tracking the loss
        self.compiled_loss(y, y_pred)
        # Update the metrics
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

def create_model(params_dict):
    #unpack relevant dictionary elements
    output_file = params_dict['output_file']
    #save model image
    save_model_image(params_dict)
    #create model
    model, params_dict = make_model_using_params_dict(params_dict)
    #load optimizer, loss function, metrics
    model_optimizer, model_loss, model_metrics = load_optimizer_loss_metrics(params_dict)
    #compile fresh model
    model.compile(optimizer=model_optimizer, loss=model_loss, metrics=model_metrics)
    #instantiate checkpoint manager
    _, manager = make_checkpoint_manager(model, model_optimizer, params_dict)
    with open(output_file, 'a') as f:
        f.write('Fresh model created successfully! \n')
    return model, manager

def load_my_model(params_dict, train_mode=False):
    #unpack relevant dictionary elements
    model_config_save_path, levels, custom_fit, mixed_precision, reload_model, output_file = unpack_keys(params_dict)
    #load config of saved model
    with open(model_config_save_path, 'rb') as f:
        config = pickle.load(f)
    #set precision type:
    set_precision_type(params_dict, train_mode)
    #instaniate model
    params_dict_from_saved_model = config['params_dict']
    operations_dictionary_from_saved_model = config['operations_dictionary']
    if train_mode == False:
        #prediction mode so simply re-use saved config
        current_params_dict = params_dict_from_saved_model
    else:
        #reloading previous model for fine-tuning so use new config settings
        current_params_dict = params_dict
        current_params_dict['blocks_per_encoder_level'] = params_dict_from_saved_model['blocks_per_encoder_level']
        current_params_dict['blocks_per_decoder_level'] = params_dict_from_saved_model['blocks_per_decoder_level']
        current_params_dict['dense_growth_rate_per_encoder_level'] = params_dict_from_saved_model['dense_growth_rate_per_encoder_level']
        current_params_dict['dense_growth_rate_per_decoder_level'] = params_dict_from_saved_model['dense_growth_rate_per_decoder_level']
        if current_params_dict['dropblock_params'] == None:
            current_params_dict['dropblock_params'] = [None]*levels
    #inputs of model
    inputs = get_input_layer(params_dict, current_params_dict['patch_size'], current_params_dict['num_inputs'], train_mode=train_mode)
    #outputs of model
    if 'network_name' not in params_dict:
        params_dict['network_name'] = 'Unet'
        current_params_dict['network_name'] = 'Unet'
    NetworkName = globals()[current_params_dict['network_name']]
    outputs = NetworkName(operations_dictionary_from_saved_model, current_params_dict)(inputs)
    #instantiate model
    if train_mode == False or custom_fit[0] == False:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
        model = CustomModel(mixed_precision, custom_fit, inputs=inputs, outputs=outputs)
    #load optimizer, loss function, metrics
    model_optimizer, model_loss, model_metrics = load_optimizer_loss_metrics(current_params_dict)
    #compile fresh model
    model.compile(optimizer=model_optimizer, loss=model_loss, metrics=model_metrics)
    #instantiate checkpoint manager
    ckpt, manager = make_checkpoint_manager(model, model_optimizer, params_dict)
    #load saved weights/optimizer for model
    if train_mode == False:
        ckpt.restore(manager.latest_checkpoint).expect_partial()
    else:
        ckpt.restore(manager.latest_checkpoint)
    if reload_model[1] == False:
        #recompile the model using a fresh optimizer to erase the saved optimizer state
        model.compile(optimizer=model_optimizer, loss=model_loss, metrics=model_metrics)
        with open(output_file, 'a') as f:
            f.write('Model weights loaded successfully. Recompiling with fresh optimizer state! \n')
    else:
        with open(output_file, 'a') as f:
            f.write('Model weights and optimizer state loaded successfully! \n')
    return model, manager, ckpt

def make_model_using_params_dict(params_dict, train_mode=True):
    #unpack relevant dictionary elements
    block_type, patch_size, pooling, normalization, activation, regularization_values, blocks_per_level, levels, dense_growth_rate_per_level, dropblock_params, num_inputs, custom_fit, mixed_precision = unpack_keys(params_dict)
    #set precision type:
    set_precision_type(params_dict)
    #choose block type
    CustomBlock = globals()[block_type]
    #choose either 2D or 3D operations based on patch size
    ndims = len(patch_size)
    ConvOperation = getattr(tf.keras.layers, 'Conv%dD' % ndims)
    ConvTransposeOperation = getattr(tf.keras.layers, 'Conv%dDTranspose' % ndims)
    UpsamplingOperation = getattr(tf.keras.layers, 'UpSampling%dD' % ndims)
    SpatialDropoutOperation = getattr(tf.keras.layers, 'SpatialDropout%dD' % ndims)
    #choose pooling operation (unless using strided convolution or projection shortcut)
    try:
        PoolOperation = getattr(tf.keras.layers, pooling + '%dD' % ndims)
    except:
        PoolOperation = None
    #choose proper normalization (if normalization is not in main TF library, check add-on library)
    try:
        NormalizationOperation = getattr(tf.keras.layers, normalization[0])
    except:
        NormalizationOperation = getattr(tfa_layers, normalization[0])
    #choose proper activation (if activation is not in layers, check activation)
    try:
        ActivationOperation = getattr(tf.keras.layers, activation[0])
    except:
        ActivationOperation = getattr(tf.keras.activations, activation[0])
    #optional AveragePooling operation used in projection shortcut
    ShortcutPooling = getattr(tf.keras.layers, 'AveragePooling%dD' % ndims)
    #optional global pooling operations for SK/CBAM blocks
    GlobalAveragePoolOperation = getattr(tf.keras.layers,'GlobalAveragePooling%dD' % ndims)
    GlobalMaxPoolOperation = getattr(tf.keras.layers,'GlobalMaxPool%dD' % ndims)
    #dense layer
    DenseOperation = getattr(tf.keras.layers, 'Dense')
    #L1/L2 regularization for conv operations
    RegularizationTerm = tf.keras.regularizers.l1_l2(*regularization_values)
    #get blocks per level for encoder and decoder
    if not isinstance(blocks_per_level, list):
        blocks_per_encoder_level = [blocks_per_level] * levels
        blocks_per_decoder_level = [blocks_per_level] * (levels - 1)
    else:
        blocks_per_encoder_level = blocks_per_level[:levels]
        blocks_per_decoder_level = blocks_per_level[levels:]
    if not isinstance(dense_growth_rate_per_level, list):
        dense_growth_rate_per_encoder_level = [dense_growth_rate_per_level] * levels
        dense_growth_rate_per_decoder_level = [dense_growth_rate_per_level] * (levels - 1)
    else:
        dense_growth_rate_per_encoder_level = dense_growth_rate_per_level[:levels]
        dense_growth_rate_per_decoder_level = dense_growth_rate_per_level[levels:]
    params_dict['blocks_per_encoder_level'] = blocks_per_encoder_level
    params_dict['blocks_per_decoder_level'] = blocks_per_decoder_level
    params_dict['dense_growth_rate_per_encoder_level'] = dense_growth_rate_per_encoder_level
    params_dict['dense_growth_rate_per_decoder_level'] = dense_growth_rate_per_decoder_level
    if dropblock_params == None:
        params_dict['dropblock_params'] = [None]*levels
    #place all operations into dictionary
    operations_dictionary = {'CustomBlock': CustomBlock, 'ConvOperation': ConvOperation, 'ConvTransposeOperation': ConvTransposeOperation, 'PoolOperation': PoolOperation, 'UpsamplingOperation': UpsamplingOperation, 'NormalizationOperation': NormalizationOperation, 'ActivationOperation': ActivationOperation, 'ShortcutPooling': ShortcutPooling, 'GlobalAveragePoolOperation': GlobalAveragePoolOperation, 'GlobalMaxPoolOperation': GlobalMaxPoolOperation, 'DenseOperation': DenseOperation, 'RegularizationTerm': RegularizationTerm, 'SpatialDropoutOperation': SpatialDropoutOperation}
    #inputs of model
    inputs = get_input_layer(params_dict, patch_size, num_inputs, train_mode=train_mode)
    #output of model
    if 'network_name' not in params_dict:
        params_dict['network_name'] = 'Unet'
    NetworkName = globals()[params_dict['network_name']]
    outputs = NetworkName(operations_dictionary, params_dict)(inputs)
    #instantiate model
    if custom_fit[0] == False:
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
        model = CustomModel(mixed_precision, custom_fit, inputs=inputs, outputs=outputs)
    return model, params_dict

def get_input_layer(params_dict, patch_size, num_inputs, train_mode=True):
    batch_size, adaptive_full_image_patching = unpack_keys(params_dict)
    if train_mode == True:
        batch_size_model = None
    else:
        batch_size_model = batch_size[2]
    #input of model
    if any(adaptive_full_image_patching):
        input_shape = (*[None]*len(patch_size), num_inputs)
    else:
        input_shape = (*patch_size, num_inputs)
    return tf.keras.Input(shape=input_shape, batch_size=batch_size_model)

def set_precision_type(params_dict, train_mode=True):
    mixed_precision, enable_xla = unpack_keys(params_dict)
    # enable XLA only if training
    if train_mode == True:
        if enable_xla == True:
            tf.config.optimizer.set_jit(True)
    # enable mixed precision
    if mixed_precision == True:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    else:
        tf.keras.mixed_precision.experimental.set_policy('float32')

def load_optimizer_loss_metrics(params_dict):
    optimizer, loss, metrics = unpack_keys(params_dict)
    try:
        model_optimizer = getattr(tf.keras.optimizers, optimizer[0])(*optimizer[1:])
    except:
        model_optimizer = getattr(tfa_optimizers, optimizer[0])(*optimizer[1:])
    #initialize parameters for loss functions
    loss_functions.LossParameters(params_dict)
    model_loss = getattr(loss_functions, loss)
    model_metrics = [getattr(loss_functions, x) for x in metrics]
    return model_optimizer, model_loss, model_metrics

def make_checkpoint_manager(model, model_optimizer, params_dict):
    model_weights_save_path, max_number_checkpoints_keep = unpack_keys(params_dict)
    ckpt = tf.train.Checkpoint(model=model, optimizer=model_optimizer)
    manager = tf.train.CheckpointManager(ckpt, model_weights_save_path, max_to_keep=max_number_checkpoints_keep)
    return ckpt, manager

def save_model_image(params_dict):
    temp_params_dict = deepcopy(params_dict)
    #replace normalization with BN to simplify model pruning
    temp_params_dict['normalization'] = ['BatchNormalization', False] 
    #replace activation with ReLU to simplify model pruning
    temp_params_dict['activation'] = ['ReLU']
    #replace mixed_precision variables to simplify model pruning
    temp_params_dict['enable_xla'] = False
    temp_params_dict['mixed_precision'] = False
    temp_params_dict['custom_fit'][0] = False
    #ensure input placeholder does not use None values to get tensor shapes printed
    temp_params_dict['adaptive_full_image_patching'] = [False, False]
    with tf.compat.v1.Session() as sess:
        with tf.Graph().as_default() as tf_graph:
            #create model
            model, temp_params_dict = make_model_using_params_dict(temp_params_dict, train_mode=False)
            # Build HiddenLayer graph
            hl_graph = hl.build_graph(tf_graph, transforms=get_transforms(temp_params_dict))
    hl_graph.save(temp_params_dict['model_image_save_path'])

def get_transforms(temp_params_dict):
    transforms = [
        # prune graph to make it easier to read
        ht.Prune("Cast"),
        ht.PruneBranch("Add > Rsqrt"),
        ht.PruneBranch("Mul > Sub"),
        # Fold remaining parts of BN into one block
        ht.Fold("Mul > Add", "Norm", "Norm"),
        # Fold Upsample into one block
        ht.Fold("((Shape > Mul) | (ExpandDims > Tile)) > Reshape > ((Shape > Mul) | (ExpandDims > Tile)) > Reshape > ((Shape > Mul) | (ExpandDims > Tile)) > Reshape > Pad", "Upsample0"),
        ht.Fold("Split > Concat > Split > Concat > Split > Concat > Pad", "Upsample0"),
        ht.Fold("Upsample0 > Conv", "Upsample1", "Upsample"),
        ht.Fold("Conv > Upsample1", "Upsample2", "Conv Upsample"),
        ht.Fold("Upsample2 > Norm > Relu", "Upsample3", "Conv Upsample Norm Act"),
        ht.Fold("Upsample1 > Norm > Relu", "Upsample4", "Upsample Norm Act"),
        #Fold Deconvolution into one block
        ht.Fold("StridedSlice > Mul", "Deconv0"),
        ht.Fold("(Deconv0 | StridedSlice | Deconv0 | Deconv0) > Pack", "Deconv1"),
        ht.Fold("Deconv0 > Pack", "Deconv_1"),
        ht.Fold("Deconv0 > Deconv_1", "Deconv_2"),
        ht.Fold("Deconv0 > Deconv_2", "Deconv_3"),
        ht.Fold("StridedSlice > Deconv_3", "Deconv_4"),
        ht.Fold("Shape > Deconv1", "Deconv2"),
        ht.Fold("Deconv2 > ConvBackpropInput", "Deconv3", "Deconvolution"),
        ht.Fold("Deconv_4 > ConvBackpropInput", "Deconv3", "Deconvolution")]
    # Fold Conv, BN, and Relu together if they come together
    if temp_params_dict['preactivation'][0] == False:
        temp_transform = [
            ht.Fold("Conv > Norm > Relu", "ConvNormRelu", "Conv Norm Act"),
            ht.Fold("Deconv3 > Norm > Relu", "DeConvNormRelu", "DeConv Norm Act")]
        transforms.extend(temp_transform)
    else:
        temp_transform = [
            ht.Fold("Norm > Relu > Conv", "NormReluConv",  "Norm Act Conv"),
            ht.Fold("Norm > Relu > Deconv3", "NormReluDeConv",  "Norm Act DeConv"),
            ht.Fold("Norm > Relu", "NormRelu",  "Norm Act"),
            ht.Fold("NormRelu > MaxPool > Conv", "NormReluMaxPoolConv", "Norm Act MaxPool Conv"),
            ht.Fold("NormRelu > Upsample2", "NormReluConvUpsample", "Norm Act Conv Upsample")]
        transforms.extend(temp_transform)
    #Fold basic residual block
    if temp_params_dict['block_type'] == 'Residual_Block':
        temp_transform = [
            ht.Fold("((ConvNormRelu > Conv > Norm) | (Conv > Norm)) > Add > Relu", "BasicResidualBlockWithShortcut",  "Basic Residual Block With Shortcut"),
            ht.Fold("(ConvNormRelu > Conv > Norm) > Add > Relu", "BasicResidualBlock",  "Basic Residual Block")]
        transforms.extend(temp_transform)
    #Fold bottleneck residual block
    elif temp_params_dict['block_type'] == 'Residual_Bottleneck_Block':
        temp_transform = [
            ht.Fold("((ConvNormRelu > ConvNormRelu > Conv > Norm) | (Conv > Norm)) > Add > Relu", "ResidualBottleneckBlockWithShortcut",  "Residual Bottleneck Block With Shortcut"),
            ht.Fold("(ConvNormRelu > ConvNormRelu > Conv > Norm) > Add > Relu", "ResidualBottleneckBlock",  "Residual Bottleneck Block")]
        transforms.extend(temp_transform)
    #Fold dense block
    elif temp_params_dict['block_type'] == 'Dense_Block':
        if temp_params_dict['preactivation'][0] == False:
            temp_transform = [
                ht.Fold("ConvNormRelu > ConvNormRelu > Concat", "DenseBottleneckBlock",  "Dense Bottleneck Block"),
                ht.Fold("ConvNormRelu > Concat", "DenseBlock",  "Basic Dense Block")]
            transforms.extend(temp_transform)
        else:
            temp_transform = [
                ht.Fold("NormReluConv > NormReluConv > Concat", "PreActDenseBottleneckBlock",  "Preact Dense Bottleneck Block"),
                ht.Fold("NormReluConv > Concat", "PreActDenseBlock",  "Preact Basic Dense Block")]
            transforms.extend(temp_transform)
    #Dropblock transforms
    if np.any(temp_params_dict['dropblock_params']):
        temp_transform = [
            ht.PruneBranch("Mul > Mul > Less"),
            ht.PruneBranch("Prod > ExpandDims > ExpandDims > Sub > Add > Add > GreaterEqual"),
            ht.PruneBranch("Neg > MaxPool > Neg > Mean"),
            ht.PruneBranch("Neg > MaxPool > Neg"),
            ht.Fold("Mean > DivNoNan > Mul", "DropBlock0", "DropBlock"),
            ht.Fold("Min > DropBlock0", "DropBlock1", "DropBlock")]
    # Fold repeated nodes
    transforms.append(ht.FoldDuplicates())
    return transforms

#helper function to instantiate convolution operation
def instantiate_convolution_layer(num_filters, dilation_rate, operations_dictionary, params_dict, segmentation_convolution=False, retina_net_bias_initializer=[False, 0.5], strided_conv=False, pool=None, activation=None, custom_kernel_size=None, weight_standardization_override=False, groups=1):
    #unpack relevant dictionary elements
    weight_standardization, patch_size, filter_size_conv, padding, use_bias_on_convolutions, conv_weights_initializer = unpack_keys(params_dict)
    ConvOperation, RegularizationTerm = unpack_keys(operations_dictionary)
    if weight_standardization_override == True:
        weight_standardization = False
    if weight_standardization == True:
        kernel_constraint = CustomWeightStandardization(axis=tuple(range(0,len(patch_size)+1)))
    else:
        kernel_constraint = None
    #Conv1 filters
    if segmentation_convolution == True:
        filter_size_conv_final = 1
        if retina_net_bias_initializer[0] == False:
            return ConvOperation(filters=num_filters, kernel_size=filter_size_conv_final, dilation_rate=dilation_rate, padding=padding, use_bias=use_bias_on_convolutions, activation=activation, kernel_regularizer=RegularizationTerm, kernel_initializer=conv_weights_initializer, kernel_constraint=kernel_constraint, groups=groups)
        else:
            bias_set_value = np.array(retina_net_bias_initializer[1])
            bias_initializer_value = -np.log((1 - bias_set_value) / bias_set_value)
            return ConvOperation(filters=num_filters, kernel_size=filter_size_conv_final, dilation_rate=dilation_rate, padding=padding, use_bias=True, activation=activation, kernel_regularizer=RegularizationTerm, kernel_initializer=conv_weights_initializer, bias_initializer=tf.constant_initializer(bias_initializer_value), groups=groups)
    #strided convs
    elif strided_conv == True:
        return ConvOperation(filters=num_filters, kernel_size=filter_size_conv, strides=pool[1], dilation_rate=dilation_rate, padding=padding, use_bias=use_bias_on_convolutions, activation=activation, kernel_regularizer=RegularizationTerm, kernel_initializer=conv_weights_initializer, kernel_constraint=kernel_constraint, groups=groups)
    #custom filter size
    elif custom_kernel_size != None:
        return ConvOperation(filters=num_filters, kernel_size=custom_kernel_size, dilation_rate=dilation_rate, padding=padding, use_bias=use_bias_on_convolutions, activation=activation, kernel_regularizer=RegularizationTerm, kernel_initializer=conv_weights_initializer, kernel_constraint=kernel_constraint, groups=groups)
    else:
        return ConvOperation(filters=num_filters, kernel_size=filter_size_conv, dilation_rate=dilation_rate, padding=padding, use_bias=use_bias_on_convolutions, activation=activation, kernel_regularizer=RegularizationTerm, kernel_initializer=conv_weights_initializer, kernel_constraint=kernel_constraint, groups=groups)

#helper function to instantiate transposed convolution operation
def instantiate_transposed_convolution_layer(num_filters, kernel_size_transposed_conv, operations_dictionary, params_dict):
    #unpack relevant dictionary elements
    padding, use_bias_on_convolutions, conv_weights_initializer = unpack_keys(params_dict)
    ConvTransposeOperation, RegularizationTerm = unpack_keys(operations_dictionary)
    return ConvTransposeOperation(filters=num_filters, kernel_size=kernel_size_transposed_conv, strides=kernel_size_transposed_conv, padding=padding, use_bias=use_bias_on_convolutions, activation=None, kernel_regularizer=RegularizationTerm, kernel_initializer=conv_weights_initializer)

#helper function to instantiate proper normalization operation
def instantiate_normalization_layer(num_filters, operations_dictionary, params_dict, zero_init=False, num_groups=None, center=True):
    #unpack relevant dictionary elements
    normalization, use_scale_on_normalization, zero_init_for_residual = unpack_keys(params_dict)
    NormalizationOperation = operations_dictionary['NormalizationOperation']
    #set gamma initializer
    if zero_init == True and zero_init_for_residual == True:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()
    #if using GroupNormalization, find proper number of groups for desired group size
    if normalization[0] == 'GroupNormalization':
        num_groups = int(num_filters / normalization[1])
        if num_groups < normalization[2]:
            num_groups == normalization[2]
        return NormalizationOperation(groups=num_groups, scale=use_scale_on_normalization, center=center, gamma_initializer=gamma_initializer, epsilon=0.00001)
    elif normalization[0] == 'BatchNormalization' and normalization[1] == False:
        return NormalizationOperation(scale=use_scale_on_normalization, center=center, gamma_initializer=gamma_initializer, trainable=False, epsilon=0.00001)
    else:
        return NormalizationOperation(scale=use_scale_on_normalization, center=center, gamma_initializer=gamma_initializer, epsilon=0.00001)

#helper function to instantiate proper activation operation
def instantiate_activation_layer(operations_dictionary, params_dict):
    #unpack relevant dictionary elements
    activation = params_dict['activation']
    ActivationOperation = operations_dictionary['ActivationOperation']
    #if using LeakyReLU or ELU, set alpha parameter for slope of negative values activation
    if activation[0] == 'LeakyReLU' or activation[0] == 'ELU':
        return ActivationOperation(alpha=activation[1])
    elif activation[0] == 'swish':
        return ActivationOperation
    else:
        return ActivationOperation()

#helper function to instantiate pooling operation
def instantiate_pooling_layer(pool, operations_dictionary, params_dict, num_filters=None):
    #unpack relevant dictionary elements
    padding, pooling, anti_aliased_pooling = unpack_keys(params_dict)
    ConvOperation, PoolOperation = unpack_keys(operations_dictionary)
    #only create pooling operation if necessary
    if pool[0] == True:
        if pooling == 'AveragePooling' or (pooling == 'MaxPool' and anti_aliased_pooling[0] == False):
            return [PoolOperation(pool_size=pool[1], padding=padding)]
        elif anti_aliased_pooling[0] == True:
            smoothing_factor = 1. / np.product(pool[1])
            if anti_aliased_pooling[1] == True and len(np.unique(pool[1])) == 1:
                filt = np.array([1,3,3,1])
                filt = filt[:, np.newaxis] * filt[np.newaxis, :]
                filt = filt[:, np.newaxis] * filt[np.newaxis, :]
                filt = filt / np.sum(filt)
                smoothing_factor = np.tile(filt[...,np.newaxis, np.newaxis], (1,1,1,1,num_filters))
                kernel_size = [4,4,4]
            else:
                kernel_size = pool[1]
            blur_pool_operation = ConvOperation(filters=num_filters, kernel_size=kernel_size, strides=pool[1], groups=num_filters, padding='same', use_bias=False, kernel_initializer=tf.constant_initializer(smoothing_factor), trainable=False)
            if pooling == 'MaxPool':
                maxpool_operation = PoolOperation(pool_size=pool[1], padding=padding, strides=1)
                return [maxpool_operation, blur_pool_operation]
            elif pooling == 'StridedConvolution':
                return [blur_pool_operation]
        else:
            return [PoolOperation(pool_size=pool[1], padding=padding)]
    else:
        return [None]

#helper function to instantiate upsampling operation
def instantiate_upsampling_layer(kernel_size_upsampling, num_filters, operations_dictionary, params_dict):
    #unpack relevant dictionary elements
    upsampling, anti_aliased_pooling = unpack_keys(params_dict)
    UpsamplingOperation, ConvOperation = unpack_keys(operations_dictionary)
    upsampling_operations_list = []
    #generic upsampling
    upsampling_operations_list.append(UpsamplingOperation(size=kernel_size_upsampling))
    #smooth with fixed kernel
    smoothing_factor = 1. / np.product(kernel_size_upsampling)
    #smooth with depthwise convolution to ensure output channels are not jointly smoothed
    if upsampling[1] == True:
        if anti_aliased_pooling[1] == True and len(np.unique(kernel_size_upsampling)) == 1:
            filt = np.array([1,3,3,1])
            filt = filt[:, np.newaxis] * filt[np.newaxis, :]
            filt = filt[:, np.newaxis] * filt[np.newaxis, :]
            filt = filt / np.sum(filt)
            smoothing_factor = np.tile(filt[...,np.newaxis, np.newaxis], (1,1,1,1,num_filters))
            kernel_size = [4,4,4]
        else:
            kernel_size = kernel_size_upsampling
        upsampling_operations_list.append(ConvOperation(filters=num_filters, kernel_size=kernel_size, groups=num_filters, padding='valid', use_bias=False, kernel_initializer=tf.constant_initializer(smoothing_factor), trainable=False))
    else:
        #use non-grouped convolution implementation, which will be faster than depthwise convolution for large networks
        init_upsample = [smoothing_factor if i == j else 0 for i in range(0,num_filters) for j in range(0,num_filters)]*np.product(kernel_size_upsampling)
        upsampling_operations_list.append(ConvOperation(filters=num_filters, kernel_size=kernel_size_upsampling, padding='valid', use_bias=False, kernel_initializer=tf.constant_initializer(init_upsample), trainable=False))
    return upsampling_operations_list

#helper function to instantiate transposed convolution operation
def instantiate_dense_layer(units, activation, operations_dictionary, params_dict, use_bias=True, retina_net_bias_initializer=[False, 0.5]):
    #unpack relevant dictionary elements
    conv_weights_initializer = params_dict['conv_weights_initializer']
    DenseOperation, RegularizationTerm = unpack_keys(operations_dictionary)
    if retina_net_bias_initializer[0] == False:
        return DenseOperation(units, use_bias=use_bias, activation=activation, kernel_regularizer=RegularizationTerm, kernel_initializer=conv_weights_initializer)
    else:
        bias_set_value = np.array(retina_net_bias_initializer[1])
        bias_initializer_value = -np.log((1 - bias_set_value) / bias_set_value)
        return DenseOperation(units, use_bias=use_bias, activation=activation, kernel_regularizer=RegularizationTerm, kernel_initializer=conv_weights_initializer, bias_initializer=tf.constant_initializer(bias_initializer_value))

#helper function to calculate number of groups given the number of filters
def calculate_group_convolution_parameters(use_grouped_convolutions, in_filters, num_filters):
    if in_filters >= (use_grouped_convolutions[1] * use_grouped_convolutions[2]):
        use_grouped_convolutions[1] = in_filters // use_grouped_convolutions[2]
    if use_grouped_convolutions[1] > in_filters:
        use_grouped_convolutions[1] = in_filters
    num_groups = in_filters // use_grouped_convolutions[1]
    out_filters = num_filters // num_groups
    return num_groups, out_filters

#helper function to implement dropblock
def dropblock(x, dropblock_params, operations_dictionary, params_dict, current_prob=None):
    mixed_precision = params_dict['mixed_precision']
    SpatialDropoutOperation = operations_dictionary['SpatialDropoutOperation']
    keep_prob = dropblock_params[0]
    if tf.is_tensor(current_prob):
        keep_prob = current_prob
    dropblock_size = dropblock_params[1]
    #get shape
    layer_shape = x.get_shape().as_list()
    #check if we are using Spatial Dropout instead of DropBlock
    if len(dropblock_params) == 4 and dropblock_params[3] == True:
        drop_rate = 1.0 - keep_prob
        if mixed_precision == True:
            drop_rate = tf.cast(drop_rate, tf.float16)
        else:
            drop_rate = tf.cast(drop_rate, tf.float32)
        x = SpatialDropoutOperation(drop_rate)(x)
        return x
    #only can use dropblock if know exact size of feature maps
    if all(layer_shape) == False and len(dropblock_params) == 2:
        return x
    elif all(layer_shape) == False and len(dropblock_params) == 3:
        layer_shape[1:-1] = dropblock_params[2]
        layer_shape[0] = params_dict['batch_size'][0]
    feature_map_dims = layer_shape[1:-1]
    min_dim = min(feature_map_dims)
    ndims = len(feature_map_dims)
    dropblock_size = min(dropblock_size, min_dim)
    #seed_drop_rate is the gamma parameter of DropBlock
    seed_drop_rate = (1.0 - keep_prob) * np.prod(feature_map_dims) / dropblock_size**ndims / np.prod([feature_map_dim - dropblock_size + 1 for feature_map_dim in feature_map_dims])
    #forces the block to be inside the feature map
    range_meshgrid = [tf.range(feature_map_dim) for feature_map_dim in feature_map_dims]
    meshgrid_values = tf.meshgrid(*range_meshgrid, indexing='ij')
    #handle even and odd sized blocks slightly differently
    if dropblock_size % 2 == 0:
        temp_locs = [tf.logical_and(vals >= (dropblock_size // 2), vals <= (feature_map_dim - (dropblock_size // 2))) for vals, feature_map_dim in zip(meshgrid_values,feature_map_dims)]
    else:
        temp_locs = [tf.logical_and(vals >= (dropblock_size // 2), vals < (feature_map_dim - (dropblock_size // 2))) for vals, feature_map_dim in zip(meshgrid_values,feature_map_dims)]
    if mixed_precision == True:
        valid_block_center = tf.reduce_prod(tf.cast(tf.stack(temp_locs), tf.float16), axis=0)
        randnoise = tf.random.uniform(layer_shape, dtype=tf.float16)
        seed_drop_rate = tf.cast(seed_drop_rate, tf.float16)
    else:
        valid_block_center = tf.reduce_prod(tf.cast(tf.stack(temp_locs), tf.float32), axis=0)
        randnoise = tf.random.uniform(layer_shape, dtype=tf.float32)
        seed_drop_rate = tf.cast(seed_drop_rate, tf.float32)
    valid_block_center = tf.expand_dims(tf.expand_dims(valid_block_center, 0), -1)
    block_pattern = (1 - valid_block_center + (1 - seed_drop_rate) + randnoise) >= 1
    if mixed_precision == True:
        block_pattern = tf.cast(block_pattern, dtype=tf.float16)
    else:
        block_pattern = tf.cast(block_pattern, dtype=tf.float32)
    #figure out which regions get dropped
    if np.all([dropblock_size == feature_map_dim for feature_map_dim in feature_map_dims]):
        if ndims == 2:
            block_pattern = tf.reduce_min(block_pattern, axis=[1, 2], keepdims=True)
        else:
            block_pattern = tf.reduce_min(block_pattern, axis=[1, 2, 3], keepdims=True)
    else:
        ksize = [1, *[dropblock_size]*ndims, 1]
        block_pattern = -tf.nn.max_pool(-block_pattern, strides=1, ksize=ksize, padding='SAME')
    #normalize
    percent_ones = tf.reduce_mean(tf.cast(block_pattern, tf.float32))
    x = tf.math.divide_no_nan(x, tf.cast(percent_ones, x.dtype)) * tf.cast(block_pattern, x.dtype)
    return x

#helper function to use normalization layers
def apply_normalization(x, normalization_operation, normalization_params, training):
    if normalization_params[0] == 'BatchNormalization':
        if normalization_params[1] == False:
            return normalization_operation(x, training=False)
        else:
            return normalization_operation(x, training=training)
    else:
        return normalization_operation(x)

#helper class to add weight standardization to kernels (make all kernels mean zero standard deviation one)
class CustomWeightStandardization(tf.keras.constraints.Constraint):
    def __init__(self, axis=0):
        self.axis = axis
    #
    def __call__(self, w):
        mean = tf.math.reduce_mean(w, axis=self.axis, keepdims=True)
        std = tf.math.sqrt(tf.math.reduce_variance(w, axis=self.axis, keepdims=True) + tf.keras.backend.epsilon()) + 0.00001
        return (w - mean) / std
    #
    def get_config(self):
        return {'axis': self.axis}