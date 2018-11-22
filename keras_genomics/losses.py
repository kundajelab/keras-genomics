from __future__ import division, absolute_import
from keras import backend as K


def ambig_binary_crossentropy(y_true,y_pred):
        non_ambig = K.cast((y_true > -0.5),'float32')
        return K.mean(K.binary_crossentropy(y_true, y_pred)
                      *non_ambig, axis=-1)


def regression_and_classification(y_true, y_pred):

    #if y_true is 0, treated as a negative
    # if y_true > 1, treated as a positive
    # nothing in between 

    y_true_binarized = K.cast(K.greater_equal(y_true, 1.0), K.floatx()) 
    hard_sigmoid_pred = K.hard_sigmoid(y_pred)
    linearized_hard_sigmoid_pred = (0.2*y_pred) + 0.5
    binary_crossentropy_loss = K.mean(K.binary_crossentropy(
                                        output=hard_sigmoid_pred,
                                        target=y_true_binarized), axis=-1)
    mse_loss = K.mean(2.71*K.square(linearized_hard_sigmoid_pred - y_true)
                          *K.cast(K.greater(y_true, 1.0), K.floatx()),
                      axis=-1)
    return binary_crossentropy_loss + mse_loss 


def hybrid_bce_loss(y_true,y_pred):
    #y_true and y_pred are structured as follows:
    #first dimension is sample,
    #second dimension has length 2; first entry in y_pred is the *logit*
    # of the sigmoid (we will put it through the sigmoid in the loss func and
    # then apply binary crossentropy)
    # second entry in y_pred is the *logit* for the IR normalized regression
    # value(we will apply a bce error loss to this, but only for the positives)

    binary_task_predicted_logits = y_pred[:,0:1]
    ir_task_predicted_logits = y_pred[:,1:2]
    binary_y_true = y_true[:,0:1]
    ir_y_true = y_true[:,1:2]
    binary_task_crossentropy_loss = K.mean(
        K.binary_crossentropy(
            target=binary_y_true,
            output=binary_task_predicted_logits,
            from_logits=True),
        axis=-1)
    ir_weight = binary_y_true
    ir_task_crossentropy_loss = K.mean(
        K.binary_crossentropy(
            target=ir_y_true,
            output=ir_task_predicted_logits,
            from_logits=True)*ir_weight,
        axis=-1)
    return binary_task_crossentropy_loss + ir_task_crossentropy_loss


def hybrid_loss(y_true,y_pred):
    #y_true and y_pred are structured as follows:
    #first dimension is sample,
    #second dimension has length 2; first entry in y_pred is the *logit*
    # of the sigmoid (we will put it through the sigmoid in the loss func and
    # then apply binary crossentropy)
    # second entry in y_pred is the predicted value for the regression (we
    # will apply a mean-squared error loss to this)

    binary_task_predicted_logits = y_pred[:,0:1]
    binary_y_true = y_true[:,0:1]
    regression_y_true = y_true[:,1:2]
    regression_task_predictions = y_pred[:,1:2]
    binary_task_crossentropy_loss = K.mean(K.binary_crossentropy(
        target=binary_y_true, output=binary_task_predicted_logits,
        from_logits=True),axis=-1)
    regression_weight = binary_y_true
    regression_loss = K.mean(K.square(regression_task_predictions -
                                      regression_y_true)*regression_weight,
                             axis=-1)
    
    return binary_task_crossentropy_loss + regression_loss


def weighted_regression(y_true,y_pred):
    #y_true and y_pred are structured as follows:
    #first dimension is sample,
    #second dimension has length 2; first entry in y_pred is the *logit*
    # of the sigmoid (we will put it through the sigmoid in the loss func and
    # then apply binary crossentropy)
    # second entry in y_pred is the predicted value for the regression (we
    # will apply a mean-squared error loss to this)

    binary_task_predicted_logits = y_pred[:,0:1]
    regression_task_predictions = y_pred[:,1:2]
    binary_task_crossentropy_loss = K.mean(K.binary_crossentropy(
        target=y_true[:,0:1], output=binary_task_predicted_logits,
        from_logits=True),axis=-1)
    positives_upweight_factor = K.sum(1.0-y_true[:,0])/K.sum(y_true[:,0])
    regression_weight = y_true[:,0:1]*(positives_upweight_factor) + (1-y_true[:,0:1])*1.0
    regression_loss = K.mean(K.square(regression_task_predictions -
                                      y_true[:,1:2])*regression_weight,
                             axis=-1)
    total_bce_loss = K.sum(binary_task_crossentropy_loss)
    total_regression_loss = K.sum(regression_loss)
    upscaled_regression_loss_factor = 1.0
    #upscaled_regression_loss_factor = K.stop_gradient(K.minimum(10.0,(total_bce_loss)/(K.maximum(total_regression_loss,0.01))))
    
    return binary_task_crossentropy_loss + regression_loss*upscaled_regression_loss_factor
    
    
