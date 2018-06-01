from __future__ import absolute_import

from keras.layers.convolutional import Conv1D
from keras.utils import conv_utils
from keras import backend as K


class RevCompConv1D(Conv1D):

    def compute_output_shape(self, input_shape):
        length = conv_utils.conv_output_length(input_shape[1],
                                               self.kernel_size[0],
                                               padding=self.padding,
                                               stride=self.strides[0])
        return (input_shape[0], length, 2*self.filters)

    def call(self, inputs):
        #create a rev-comped kernel.
        #kernel shape is (width, input_channels, filters)
        #Rev comp is along both the length (dim 0) and input channel (dim 1)
        #axes; that is the reason for ::-1, ::-1 in the first and second dims.
        #The rev-comp of channel at index i should be at index -i
        #This is the reason for the ::-1 in the last dim.
        revcomp_kernel =\
            K.concatenate([self.kernel,
                           self.kernel[::-1,::-1,::-1]],axis=-1)
        if (self.use_bias):
            revcomp_bias = K.concatenate([self.bias,
                                          self.bias[::-1]], axis=-1)

        outputs = K.conv1d(inputs, revcomp_kernel,
                           strides=self.strides[0],
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate[0])

        if self.use_bias:
            outputs += K.bias_add(outputs,
                                  revcomp_bias,
                                  data_format=self.data_format)

        if (self.activation is not None):
            outputs = self.activation(outputs)
        return outputs


