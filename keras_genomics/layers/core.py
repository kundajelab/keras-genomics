from __future__ import absolute_import
from __future__ import division

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import InputSpec
from keras.engine import Layer
from keras.utils.generic_utils import get_custom_objects
from keras.layers.core import Dense


class SeparableFC(Layer):
    """A separable fully-connected NN layer
    Separable Fully Connected Layers Improve Deep Learning Models For Genomics
    https://doi.org/10.1101/146431
    # Example
        Expected usage is after a stack of convolutional layers and before
        densely connected layers
        A gist illustrating model setup is at: goo.gl/gYooaa
    # Arguments
        output_dim: the number of output neurons
        symmetric: if weights are to be symmetric along length, set to True
        smoothness_regularizer: regularization to be applied on adjacent
            weights in the length dimension of positional weights matrix
        positional_constraint: constraint to be enforced on adjacent
            weights in the length dimension of positional weights matrix
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, output_features)`.
    """
    def __init__(self, output_dim, symmetric=False,
                 smoothness_regularizer=None,
                 positional_constraint=None, **kwargs):
        super(SeparableFC, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.symmetric = symmetric
        self.smoothness_regularizer = smoothness_regularizer
        self.positional_constraint = positional_constraint

    def build(self, input_shape):
        import numpy as np
        self.original_length = input_shape[1]
        if self.symmetric is False:
            self.length = input_shape[1]
        else:
            self.odd_input_length = input_shape[1] % 2.0 == 1
            self.length = int(input_shape[1] / 2.0 + 0.5)
        self.num_channels = input_shape[2]
        limit = np.sqrt(np.sqrt(
            2.0 / (self.length * self.num_channels + self.output_dim)))
        self.W_pos = self.add_weight(
            shape=(self.output_dim, self.length),
            name='{}_W_pos'.format(self.name),
            initializer=initializers.uniform(-1 * limit, limit),
            constraint=self.positional_constraint,
            regularizer=self.smoothness_regularizer,
            trainable=True)
        self.W_chan = self.add_weight(
            shape=(self.output_dim, self.num_channels),
            name='{}_W_chan'.format(self.name),
            initializer=initializers.uniform(-1 * limit, limit),
            trainable=True)
        self.built = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def call(self, x, mask=None):
        if self.symmetric is False:
            W_pos = self.W_pos
        else:
            W_pos = K.concatenate(
                tensors=[self.W_pos,
                         self.W_pos[:, ::-1][:, (1 if self.odd_input_length else 0):]],
                axis=1)
        W_output = K.expand_dims(W_pos, 2) * K.expand_dims(self.W_chan, 1)
        W_output = K.reshape(W_output,
                             (self.output_dim, self.original_length * self.num_channels))
        x = K.reshape(x,
                      (-1, self.original_length * self.num_channels))
        output = K.dot(x, K.transpose(W_output))
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'symmetric': self.symmetric,
                  'smoothness_regularizer': self.smoothness_regularizer,
                  'positional_constraint': self.positional_constraint}
        base_config = super(SeparableFC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
get_custom_objects().update({'SeparableFC': SeparableFC})


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

