from __future__ import division, absolute_import
from keras import backend as K


def ambig_binary_crossentropy(y_true,y_pred):
        non_ambig = K.cast((y_true > -0.5),'float32')
        return K.mean(K.binary_crossentropy(y_pred, y_true)
                      *non_ambig, axis=-1)
