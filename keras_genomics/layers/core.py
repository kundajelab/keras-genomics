from __future__ import absolute_import
from __future__ import division

from keras.engine import InputSpec
from keras.layers.core import Dense
from keras import backend as K


class DenseAfterRevcompConv1D(Dense):
    '''For dense layers that follow 1D Convolutional or Pooling layers that
    have reverse-complement weight sharing
    '''

    def build(self, input_shape):
        assert len(input_shape) == 3, "layer designed to follow 1D conv/pool"
        num_chan = input_shape[-1]
        input_length = input_shape[-2]
        assert num_chan%2 == 0, "num_chan should be even; is "+str(num_chan)
        self.num_chan = num_chan
        self.input_length = input_length

        self.kernel = self.add_weight(
                    shape=(int(input_length*num_chan/2), self.units),
                           initializer=self.kernel_initializer,
                           name='kernel',
                           regularizer=self.kernel_regularizer,
                           constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(
                            shape=(self.units,),
                            initializer=self.bias_initializer,
                            name='bias',
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: num_chan})
        self.built = True

    def call(self, inputs):
        kernel = K.reshape(self.kernel,
                           (self.input_length, int(self.num_chan/2),
                            self.units))
        concatenated_reshaped_kernel = K.reshape(K.concatenate(
            tensors=[kernel, kernel[::-1,::-1,:]], axis=1),
            (self.input_length*self.num_chan, self.units))
        reshaped_inputs =\
            K.reshape(inputs, (-1, self.input_length*self.num_chan))
        output = K.dot(reshaped_inputs,
                       concatenated_reshaped_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        assert input_shape[-1] is not None
        assert input_shape[-2] is not None
        output_shape = [input_shape[0], self.units]
        return tuple(output_shape)

