#TODO: Determine optimal alpha-reweighting for weighted cross entropy and for focal loss
#TODO: Add weighting to multi-class dice
#TODO: Fix sigmoid cross-entropy for categorical outputs 
#TODO: Add more custom metrics (HD95, detection rate, etc.)
#TODO: Fix classification loss(es) to work with mirrored strategy (https://www.tensorflow.org/tutorials/distribute/custom_training)

import numpy as np
import tensorflow as tf

from itertools import product
from sorcery import unpack_keys
from model_callbacks import DecayAlphaParameter

#binary cross-entropy for classification using logits
def binary_cross_entropy_classification(y_true, y_pred):
    #re-weight cross-entropy loss using desired sample weighting
    if y_true.shape[-1] == 1:
        y_true_onehot = one_hot_encode(tf.squeeze(y_true, axis=-1), LossParameters.num_classes)
    else:
        y_true_onehot = one_hot_encode(y_true[...,0], LossParameters.num_classes)
    sample_weight = tf.reduce_sum(tf.multiply(y_true_onehot, LossParameters.weights[:,0]), axis=-1)
    return tf.losses.BinaryCrossentropy(from_logits=True)(y_true, y_pred, sample_weight=sample_weight)

#binary accuracy metric for classification using logits
def binary_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.metrics.binary_accuracy(y_true, y_pred, threshold=0.0))

#binary accuracy metric for classification using logits (specifically for use as validation monitor since it will average over all patient crops)
def binary_accuracy_batch(y_true, y_pred):
    y_true_batch = tf.reduce_mean(y_true, axis=0)
    y_pred_batch = tf.reduce_mean(y_pred, axis=0)
    return tf.reduce_mean(tf.metrics.binary_accuracy(y_true_batch, y_pred_batch, threshold=0.0))

#multi-class sorensen-dice coefficient metric
def dice_coef_metric(y_true, y_pred):
    #activate outputs
    if y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true[...,0], -1)
    y_true_onehot, y_pred_prob = activate_ouputs(y_true, y_pred)
    #can treat batch as a "pseudo-volume", or collect dice metrics on each volume in the batch individually
    axes_to_sum = find_axes_to_sum(y_true_onehot)
    #calculate dice metric per class
    intersection = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_prob), axis=axes_to_sum)
    union = tf.add(tf.reduce_sum(y_true_onehot, axis=axes_to_sum), tf.reduce_sum(y_pred_prob, axis=axes_to_sum))
    numerator = tf.add(tf.multiply(intersection, 2.), LossParameters.smooth)
    denominator = tf.add(union, LossParameters.smooth)
    dice_metric_per_class = tf.divide(numerator, denominator)
    #return average dice metric over classes (choosing to use or not use the background class)
    return calculate_final_dice_metric(dice_metric_per_class)

#multi-class hard sorensen-dice coefficient metric (only should be used as metric, not as loss)
def hard_dice_coef_metric(y_true, y_pred):
    #activate outputs
    if y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true[...,0], -1)
    y_true_onehot, y_pred_prob = activate_ouputs(y_true, y_pred)
    #can treat batch as a "pseudo-volume", or collect dice metrics on each volume in the batch individually
    axes_to_sum = find_axes_to_sum(y_true_onehot)
    #argmax to find predicted class and then one-hot the predicted vector
    y_pred_onehot = one_hot_encode(tf.math.argmax(y_pred_prob, axis=-1), LossParameters.num_classes)
    #calculate dice metric per class
    intersection = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_onehot), axis=axes_to_sum)
    union = tf.add(tf.reduce_sum(y_true_onehot, axis=axes_to_sum), tf.reduce_sum(y_pred_onehot, axis=axes_to_sum))
    numerator = tf.multiply(intersection, 2.)
    denominator = union
    dice_metric_per_class = tf.divide(numerator, denominator)
    #replace any NaN values with 1.0 (NaN only occurs when both the ground truth predicted label is empty, which should give a true dice score of 1.0)
    dice_metric_per_class = tf.where(tf.math.is_nan(dice_metric_per_class), tf.ones_like(dice_metric_per_class), dice_metric_per_class)
    #return average dice metric over classes (choosing to use or not use the background class)
    return calculate_final_dice_metric(dice_metric_per_class)

#multi-class sorensen-dice coefficient loss
def dice_coef_loss(y_true, y_pred):
    return tf.subtract(1., dice_coef_metric(y_true, y_pred))

#multi-class jaccard similarity coefficient metric
def jaccard_coef_metric(y_true, y_pred):
    #activate outputs
    if y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true[...,0], -1)
    y_true_onehot, y_pred_prob = activate_ouputs(y_true, y_pred)
    #can treat batch as a "pseudo-volume", or collect dice metrics on each volume in the batch individually
    axes_to_sum = find_axes_to_sum(y_true_onehot)
    intersection = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_prob), axis=axes_to_sum)
    union = tf.add(tf.reduce_sum(y_true_onehot, axis=axes_to_sum), tf.reduce_sum(y_pred_prob, axis=axes_to_sum))
    numerator = tf.add(intersection, LossParameters.smooth)
    denominator = tf.add(tf.subtract(union, intersection), LossParameters.smooth)
    jaccard_metric_per_class = tf.divide(numerator, denominator)
    #return average jaccard metric over classes (choosing to use or not use the background class)
    return calculate_final_dice_metric(jaccard_metric_per_class)

#jaccard similarity coefficient loss
def jaccard_coef_loss(y_true, y_pred):
    return tf.subtract(1., jaccard_coef_metric(y_true, y_pred))

#weighted categorical cross entropy loss
def weighted_cross_entropy_loss(y_true, y_pred):
    if y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true[...,0], -1)
    y_true_onehot, y_pred_prob, cross_entropy_matrix = cross_entropy_loss_matrix(y_true, y_pred)
    return tf.multiply(weight_matrix(y_true_onehot, y_pred_prob), cross_entropy_matrix)

#weighted binary cross entropy with boundary loss
def weighted_boundary_loss(y_true, y_pred):
    return tf.multiply(y_true[...,1], weighted_cross_entropy_loss(y_true, y_pred))

#categorical focal loss
def focal_loss(y_true, y_pred):
    if y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true[...,0], -1)
    y_true_onehot, y_pred_prob, cross_entropy_matrix = cross_entropy_loss_matrix(y_true, y_pred)
    alpha_term = weight_matrix(y_true_onehot, y_pred_prob)
    gamma_term = focal_weight(y_true_onehot, y_pred_prob)
    return tf.multiply(tf.multiply(alpha_term, gamma_term), cross_entropy_matrix)

#combined dice loss and weighted cross entropy loss
def joint_dice_cross_entropy_loss(y_true, y_pred):
    loss_contribution1 = tf.multiply(DecayAlphaParameter.alpha1, dice_coef_loss(y_true, y_pred))
    loss_contribution2 = tf.multiply(DecayAlphaParameter.alpha2, weighted_cross_entropy_loss(y_true, y_pred))
    return tf.add(loss_contribution1, loss_contribution2)

#combined dice loss and binary boundary loss
def joint_dice_boundary_loss(y_true, y_pred):
    loss_contribution1 = tf.multiply(DecayAlphaParameter.alpha1, dice_coef_loss(y_true, y_pred))
    loss_contribution2 = tf.multiply(DecayAlphaParameter.alpha2, weighted_boundary_loss(y_true, y_pred))
    return tf.add(loss_contribution1, loss_contribution2)

#combined dice loss and focal loss
def joint_dice_focal_loss(y_true, y_pred):
    loss_contribution1 = tf.multiply(DecayAlphaParameter.alpha1, dice_coef_loss(y_true, y_pred))
    loss_contribution2 = tf.multiply(DecayAlphaParameter.alpha2, focal_loss(y_true, y_pred))
    return tf.add(loss_contribution1, loss_contribution2)

#combined dice loss and focal loss using adaptive weighting based on current dice score
def adaptive_dice_focal_loss(y_true, y_pred):
    batch_dice_loss = dice_coef_loss(y_true, y_pred)
    batch_focal_loss = focal_loss(y_true, y_pred)
    weighting_value = tf.divide(1., tf.add(1., tf.math.exp(tf.multiply(-50., tf.subtract(batch_dice_loss, .35)))))
    loss_contribution1 = tf.multiply(weighting_value, batch_dice_loss)
    loss_contribution2 = tf.multiply(tf.subtract(1., weighting_value), batch_focal_loss)
    return tf.add(loss_contribution1, loss_contribution2)

#dice metric for brats segmentation (converts sparse ground truth into overlapping set of regions to be used with sigmoid)
def brats_dice_coef_metric(y_true, y_pred):
    #activate outputs
    y_pred_prob = sigmoid_probability(y_pred)
    if y_true.shape[-1] == 1:
        y_true_onehot_orig = one_hot_encode(tf.squeeze(y_true, axis=-1), 4)
    else:
        y_true_onehot_orig = one_hot_encode(y_true[...,0], 4)
    #convert ground truth to proper region labels
    background = y_true_onehot_orig[...,0]
    enhancing_tumor = y_true_onehot_orig[...,3]
    tumor_core = tf.add(y_true_onehot_orig[...,1], enhancing_tumor)
    whole_tumor = tf.add(y_true_onehot_orig[...,2], tumor_core)
    y_true_onehot = tf.stack([background, enhancing_tumor, tumor_core, whole_tumor], axis=-1)
    #can treat batch as a "pseudo-volume", or collect dice metrics on each volume in the batch individually
    axes_to_sum = find_axes_to_sum(y_true_onehot)
    #add "fake" background channel to predicted
    y_pred_prob = tf.concat([tf.expand_dims(background, axis=-1), y_pred_prob], axis=-1)
    #calculate dice metric per class
    intersection = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_prob), axis=axes_to_sum)
    union = tf.add(tf.reduce_sum(y_true_onehot, axis=axes_to_sum), tf.reduce_sum(y_pred_prob, axis=axes_to_sum))
    numerator = tf.add(tf.multiply(intersection, 2.), LossParameters.smooth)
    denominator = tf.add(union, LossParameters.smooth)
    dice_metric_per_class = tf.divide(numerator, denominator)
    #return average dice metric over classes (choosing to use or not use the background class)
    return calculate_final_dice_metric(dice_metric_per_class)

#dice coefficient loss for use with brats dataset
def brats_dice_coef_loss(y_true, y_pred):
    return tf.subtract(1., brats_dice_coef_metric(y_true, y_pred))

#hard dice metric for brats segmentation (converts sparse ground truth into overlapping set of regions to be used with sigmoid)
def hard_brats_dice_coef_metric(y_true, y_pred):
    #activate outputs
    y_pred_prob = sigmoid_probability(y_pred)
    if y_true.shape[-1] == 1:
        y_true_onehot_orig = one_hot_encode(tf.squeeze(y_true, axis=-1), 4)
    else:
        y_true_onehot_orig = one_hot_encode(y_true[...,0], 4)
    #convert ground truth to proper region labels
    background = y_true_onehot_orig[...,0]
    enhancing_tumor = y_true_onehot_orig[...,3]
    tumor_core = tf.add(y_true_onehot_orig[...,1], enhancing_tumor)
    whole_tumor = tf.add(y_true_onehot_orig[...,2], tumor_core)
    y_true_onehot = tf.stack([background, enhancing_tumor, tumor_core, whole_tumor], axis=-1)
    #can treat batch as a "pseudo-volume", or collect dice metrics on each volume in the batch individually
    axes_to_sum = find_axes_to_sum(y_true_onehot)
    #add "fake" background channel to predicted
    y_pred_prob = tf.concat([tf.expand_dims(background, axis=-1), y_pred_prob], axis=-1)
    #calculate dice metric per class
    y_pred_binary = tf.cast(tf.greater_equal(y_pred_prob, 0.5), tf.float32)
    intersection = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_binary), axis=axes_to_sum)
    union = tf.add(tf.reduce_sum(y_true_onehot, axis=axes_to_sum), tf.reduce_sum(y_pred_binary, axis=axes_to_sum))
    numerator = tf.multiply(intersection, 2.)
    denominator = union
    dice_metric_per_class = tf.divide(numerator, denominator)
    #replace any NaN values with 1.0 (NaN only occurs when both the ground truth predicted label is empty, which should give a true dice score of 1.0)
    dice_metric_per_class = tf.where(tf.math.is_nan(dice_metric_per_class), tf.ones_like(dice_metric_per_class), dice_metric_per_class)
    #return average dice metric over classes (choosing to use or not use the background class)
    return calculate_final_dice_metric(dice_metric_per_class)

#dice metric for only brats whole tumor
def brats_region_dice_coef_metric(y_true, y_pred):
    if y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true[...,0], -1)
    enhancing_tumor = tf.cast(y_true == 3, tf.float32)
    tumor_core = tf.add(tf.cast(y_true == 1, tf.float32), enhancing_tumor)
    whole_tumor = tf.add(tf.cast(y_true == 2, tf.float32), tumor_core)
    y_true_region = tumor_core
    return dice_coef_metric(y_true_region, y_pred)

#dice loss for only brats whole tumor
def brats_region_dice_coef_loss(y_true, y_pred):
    return tf.subtract(1., brats_region_dice_coef_metric(y_true, y_pred))

#dice loss for only brats whole tumor
def brats_region_dice_and_boundary_loss(y_true, y_pred):
    loss1 = tf.subtract(1., brats_region_dice_coef_metric(y_true, y_pred))
    loss2 = weighted_boundary_loss(y_true, y_pred)
    return loss1 + loss2

#hard dice metric for only brats whole tumor
def hard_brats_region_dice_coef_metric(y_true, y_pred):
    if y_true.shape[-1] != 1:
        y_true = tf.expand_dims(y_true[...,0], -1)
    enhancing_tumor = tf.cast(y_true == 3, tf.float32)
    tumor_core = tf.add(tf.cast(y_true == 1, tf.float32), enhancing_tumor)
    whole_tumor = tf.add(tf.cast(y_true == 2, tf.float32), tumor_core)
    y_true_region = tumor_core
    return hard_dice_coef_metric(y_true_region, y_pred)

#joint dice and focal loss for brats segmentation (messy but works)
def brats_dice_and_focal_loss(y_true, y_pred):
    #activate outputs
    y_pred_prob = sigmoid_probability(y_pred)
    y_true_onehot_orig = one_hot_encode(tf.squeeze(y_true, axis=-1), 4)
    #convert ground truth to proper region labels
    background = y_true_onehot_orig[...,0]
    enhancing_tumor = y_true_onehot_orig[...,3]
    tumor_core = tf.add(y_true_onehot_orig[...,1], enhancing_tumor)
    whole_tumor = tf.add(y_true_onehot_orig[...,2], tumor_core)
    y_true_onehot = tf.stack([background, enhancing_tumor, tumor_core, whole_tumor], axis=-1)
    #can treat batch as a "pseudo-volume", or collect dice metrics on each volume in the batch individually
    axes_to_sum = find_axes_to_sum(y_true_onehot)
    #add "fake" background channel to predicted
    y_pred_prob = tf.concat([tf.expand_dims(background, axis=-1), y_pred_prob], axis=-1)
    #calculate dice metric per class
    intersection = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_prob), axis=axes_to_sum)
    union = tf.add(tf.reduce_sum(y_true_onehot, axis=axes_to_sum), tf.reduce_sum(y_pred_prob, axis=axes_to_sum))
    numerator = tf.add(tf.multiply(intersection, 2.), LossParameters.smooth)
    denominator = tf.add(union, LossParameters.smooth)
    dice_metric_per_class = tf.divide(numerator, denominator)
    #return average dice loss over classes (choosing to use or not use the background class)
    if LossParameters.dice_over_batch == False:
        dice_metric_per_class = tf.reduce_mean(dice_metric_per_class, axis=0)
    dice_loss = tf.subtract(1., dice_metric_per_class[1:])
    #loop over classes and get binary focal loss of each
    for i in range(0,3):
        y_true_class = tf.expand_dims(y_true_onehot[...,i+1], axis=-1)
        y_pred_class = tf.expand_dims(y_pred[...,i], axis=-1)
        y_pred_sigmoid = sigmoid_probability(y_pred_class)
        y_pred_class_prob = tf.concat([tf.subtract(1., y_pred_sigmoid), y_pred_sigmoid], axis=-1)
        y_true_class_onehot = one_hot_encode(tf.squeeze(y_true_class, axis=-1), 2)
        #cross-entropy
        cross_entropy_matrix = tf.losses.binary_crossentropy(y_true_class, y_pred_class, from_logits=True)
        #alpha term
        alpha_term = tf.zeros_like(y_pred_class_prob[...,0])
        y_pred_class_onehot = one_hot_encode(tf.math.argmax(y_pred_class_prob, axis=-1), 2)
        #enhancing tumor
        if i == 0:
            temp_weights = np.array([[2., 2.], [1., 1.]], dtype=np.float32)
        #tumor core
        elif i == 1:
            temp_weights = np.array([[1.875, 1.875], [1., 1.]], dtype=np.float32)
        #whole tumor
        else:
            temp_weights = np.array([[1.75, 1.75], [1., 1.]], dtype=np.float32)
        #make per-voxel weight mask given what our network predicted
        for (j,k) in product(range(0, 2), range(0, 2)):
            w = temp_weights[j][k]
            y_t = y_true_class_onehot[...,j]
            y_p = y_pred_class_onehot[...,k]
            alpha_term = tf.add(alpha_term, tf.multiply(w, tf.multiply(y_t, y_p)))
        #loss matrices for the three regions
        if i == 0:
            gamma_term = focal_weight(y_true_class_onehot, y_pred_class_prob, gamma_power=1.5)
            loss1 = tf.multiply(tf.multiply(alpha_term, gamma_term), cross_entropy_matrix)
        elif i == 1:
            gamma_term = focal_weight(y_true_class_onehot, y_pred_class_prob, gamma_power=1.375)
            loss2 = tf.multiply(tf.multiply(alpha_term, gamma_term), cross_entropy_matrix)
        else:
            gamma_term = focal_weight(y_true_class_onehot, y_pred_class_prob, gamma_power=1.25)
            loss3 = tf.multiply(tf.multiply(alpha_term, gamma_term), cross_entropy_matrix)
    focal_loss = tf.stack([loss1, loss2, loss3], axis=-1)
    #return joint (unweighted) sum of dice and focal loss per-pixel
    return tf.add(dice_loss, focal_loss)

#joint dice and boundary loss for brats segmentation (messy but works)
def brats_dice_and_boundary_loss(y_true, y_pred):
    #activate outputs
    y_pred_prob = sigmoid_probability(y_pred)
    y_true_onehot_orig = one_hot_encode(y_true[...,0], 4)
    #convert ground truth to proper region labels
    background = y_true_onehot_orig[...,0]
    enhancing_tumor = y_true_onehot_orig[...,3]
    tumor_core = tf.add(y_true_onehot_orig[...,1], enhancing_tumor)
    whole_tumor = tf.add(y_true_onehot_orig[...,2], tumor_core)
    y_true_onehot = tf.stack([background, enhancing_tumor, tumor_core, whole_tumor], axis=-1)
    #can treat batch as a "pseudo-volume", or collect dice metrics on each volume in the batch individually
    axes_to_sum = find_axes_to_sum(y_true_onehot)
    #add "fake" background channel to predicted
    y_pred_prob = tf.concat([tf.expand_dims(background, axis=-1), y_pred_prob], axis=-1)
    #calculate dice metric per class
    intersection = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_prob), axis=axes_to_sum)
    union = tf.add(tf.reduce_sum(y_true_onehot, axis=axes_to_sum), tf.reduce_sum(y_pred_prob, axis=axes_to_sum))
    numerator = tf.add(tf.multiply(intersection, 2.), LossParameters.smooth)
    denominator = tf.add(union, LossParameters.smooth)
    dice_metric_per_class = tf.divide(numerator, denominator)
    #return average dice loss over classes (choosing to use or not use the background class)
    if LossParameters.dice_over_batch == False:
        dice_metric_per_class = tf.reduce_mean(dice_metric_per_class, axis=0)
    dice_loss = tf.subtract(1., dice_metric_per_class[1:])
    #loop over classes and get binary focal loss of each
    for i in range(0,3):
        y_true_class = tf.expand_dims(y_true_onehot[...,i+1], axis=-1)
        y_pred_class = tf.expand_dims(y_pred[...,i], axis=-1)
        y_pred_sigmoid = sigmoid_probability(y_pred_class)
        y_pred_class_prob = tf.concat([tf.subtract(1., y_pred_sigmoid), y_pred_sigmoid], axis=-1)
        y_true_class_onehot = one_hot_encode(tf.squeeze(y_true_class, axis=-1), 2)
        #cross-entropy
        cross_entropy_matrix = tf.losses.binary_crossentropy(y_true_class, y_pred_class, from_logits=True)
        if i == 0:
            loss1 = tf.multiply(y_true[...,1], cross_entropy_matrix)
        if i == 1:
            #loss2 = tf.multiply(y_true[...,2], cross_entropy_matrix)
            loss2 = tf.zeros_like(y_true[...,2])
        if i == 2:
            loss3 = tf.multiply(y_true[...,3], cross_entropy_matrix)
    boundary_loss = tf.stack([loss1, loss2, loss3], axis=-1)
    #return joint (unweighted) sum of dice and boundary loss per-pixel
    return tf.add(dice_loss, boundary_loss)

#helper function to turn sparse output into one-hot encoded output
def one_hot_encode(y, num_classes):
    return tf.cast(tf.one_hot(tf.cast(y, tf.int32), num_classes), tf.float32)

#helper function to turn binary logits into probabilities (via sigmoid)
def sigmoid_probability(y):
    return tf.keras.activations.sigmoid(y)

#helper function to turn categorical logits into probabilities (via softmax)
def softmax_probability(y):
    return tf.keras.activations.softmax(y)

#helper function to compute un-weighted cross-entropy loss
def cross_entropy_loss_matrix(y_true, y_pred):
    y_true_onehot, y_pred_prob = activate_ouputs(y_true, y_pred)
    #if binary output, use from logits loss
    if LossParameters.num_classes == 2:
        cross_entropy_matrix = tf.losses.binary_crossentropy(y_true, y_pred, from_logits=True)
    else:
        #Use non-logits based loss if specifically using sigmoid for multi-class, otherwise use logits with one-hot true vector for numerical stability
        if LossParameters.use_sigmoid_for_multi_class == True:
            cross_entropy_matrix = tf.losses.categorical_crossentropy(y_true_onehot, y_pred_prob, from_logits=False)
        else:
            cross_entropy_matrix = tf.losses.categorical_crossentropy(y_true_onehot, y_pred, from_logits=True)
    return y_true_onehot, y_pred_prob, cross_entropy_matrix

#helper function to create custom re-weighting matrix used with cross entropy loss
def weight_matrix(y_true_onehot, y_pred_prob):
    #create penalty matrix: e.g. for binary task [[true negative, false positive], [false negative, true positive]]
    #re-weight loss function to apply less weight to negative class
    #apply different penalty to false positive vs false negative according to task requirements
    final_mask = tf.zeros_like(y_pred_prob[...,0])
    #argmax to find predicted class and then one-hot the predicted vector
    y_pred_onehot = one_hot_encode(tf.math.argmax(y_pred_prob, axis=-1), LossParameters.num_classes)
    #make per-voxel weight mask given what our network predicted
    for (i,j) in product(range(0, LossParameters.num_classes), range(0, LossParameters.num_classes)):
        w = LossParameters.weights[i][j]
        y_t = y_true_onehot[...,i]
        y_p = y_pred_onehot[...,j]
        final_mask = tf.add(final_mask, tf.multiply(w, tf.multiply(y_t, y_p)))
    return final_mask

#helper function to create re-weighting matrix used with focal loss
def focal_weight(y_true_onehot, y_pred_prob, gamma_power=None):
    if gamma_power == None:
        gamma_power = LossParameters.gamma
    p_t = tf.reduce_sum(tf.multiply(y_true_onehot, y_pred_prob), axis=-1)
    return tf.pow(tf.subtract(1., p_t), gamma_power)

#helper function to activate outputs (i.e. apply sigmoid/softmax to predicted and one-hot encode to ground truth)
def activate_ouputs(y_true, y_pred):
    #handle case of binary segmentation task where we are only predicting a single foreground channel
    if LossParameters.num_classes == 2:
        y_pred_sigmoid = sigmoid_probability(y_pred)
        y_pred_prob = tf.concat([tf.subtract(1., y_pred_sigmoid), y_pred_sigmoid], axis=-1)
    else:
        #if have multi-class segmentation, choose either to generate true softmax probabilities, or overlapping sigmoid probabilities
        if LossParameters.use_sigmoid_for_multi_class == True:
            y_pred_prob = sigmoid_probability(y_pred)
        else:
            y_pred_prob = softmax_probability(y_pred)
    y_true_onehot = one_hot_encode(tf.squeeze(y_true, axis=-1), LossParameters.num_classes)
    return y_true_onehot, y_pred_prob

#helper function to determine what axes to sum over for dice loss depending on if you are averaging dice across the batch or individually across patches
def find_axes_to_sum(y_true_onehot):
    #get number of dimensions (not including channel dimension)
    num_dims = len(y_true_onehot.shape)-1
    if LossParameters.dice_over_batch == True:
        axes_to_sum = range(0, num_dims)
    else:
        axes_to_sum = range(1, num_dims)
    return axes_to_sum

#helper function to average dice across batch and whether to include background channel
def calculate_final_dice_metric(dice_metric_per_class):
    if LossParameters.dice_over_batch == False:
        dice_metric_per_class = tf.reduce_mean(dice_metric_per_class, axis=0)
    if LossParameters.dice_with_background_class == True:
        return tf.reduce_mean(dice_metric_per_class)
    else:
        return tf.reduce_mean(dice_metric_per_class[1:])

#class to initialize common loss parameters
class LossParameters():
    num_classes = 2
    weights = np.ones((num_classes, num_classes), np.float32)
    gamma = 2.0
    dice_over_batch = True
    dice_with_background_class = False
    use_sigmoid_for_multi_class = False
    smooth = 1.
    def __init__(self, params_dict, smooth=1.):
        num_outputs, factor_reweight_foreground_classes, gamma, dice_over_batch, dice_with_background_class, use_sigmoid_for_multi_class = unpack_keys(params_dict)
        LossParameters.num_classes = np.maximum(num_outputs,2)
        weights = np.ones((np.repeat(LossParameters.num_classes, 2)), dtype=np.float32)
        weights = weights * np.expand_dims(np.array(factor_reweight_foreground_classes),axis=-1)
        LossParameters.weights = weights.astype(np.float32)
        LossParameters.gamma = gamma
        LossParameters.dice_over_batch = dice_over_batch
        LossParameters.dice_with_background_class = dice_with_background_class
        LossParameters.use_sigmoid_for_multi_class = use_sigmoid_for_multi_class
        LossParameters.smooth = smooth