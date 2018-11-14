from __future__ import division, absolute_import
from keras import backend as K


def ambig_binary_crossentropy(y_true,y_pred):
        non_ambig = K.cast((y_true > -0.5),'float32')
        return K.mean(K.binary_crossentropy(y_true, y_pred)
                      *non_ambig, axis=-1)


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
        from_logits=True))
    regression_loss = K.mean(K.square(regression_task_predictions -
                                      y_true[:,1:2])*y_true[:,0:1],
                             axis=-1)
    return binary_task_crossentropy_loss + regression_loss
    
    
