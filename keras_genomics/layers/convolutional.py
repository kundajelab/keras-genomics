from __future__ import absolute_import

from keras.layers.convolutional import Conv1D
from keras.utils import conv_utils
from keras import backend as K


class Conv1DTranspose(Conv1D):
    """1D analog of Conv2DTranspose
    # Arguments
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of a single integer, specifying the
            length of the 1D convolution window.
        strides: An integer or tuple/list of a single integer,
            specifying the stride of the convolution
            along the length.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        output_padding: An integer or tuple/list of a single integer,
            specifying the amount of padding along the length
            of the output tensor.
            The amount of output padding along a given dimension must be
            lower than the stride along that same dimension.
            If set to `None` (default), the output shape is inferred.
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, steps, channels)` while `"channels_first"`
            corresponds to inputs with shape
            `(batch, steps, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: an integer or tuple/list of 1 integer, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any stride value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        3D tensor with shape:
        `(batch, steps, channels)`
    # Output shape
        3D tensor with shape:
        `(batch, new_steps, filters)`
        `steps` value might have changed due to padding.
        If `output_padding` is specified:
        ```
        new_steps = ((steps - 1) * stride + kernel_size
                     - 2 * padding + output_padding)
        ```
    # References
        - [A guide to convolution arithmetic for deep learning](
           https://arxiv.org/abs/1603.07285v1)
        - [Deconvolutional Networks](
           https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    """

    def __init__(self, filters,
                 kernel_size,
                 strides=(1,),
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1,),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv1DTranspose, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 1, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                     'greater than output padding ' +
                                     str(self.output_padding))
        
        assert self.data_format=="channels_last" #for now, stick to basics 
       

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Inputs should have rank ' +
                             str(3) +
                             '; Received input shape:', str(input_shape))
        #if self.data_format == 'channels_first':
        #    channel_axis = 1
        #else:
        channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        #if self.data_format == 'channels_first':
        #    s_axis = 2
        #else:
        s_axis = 1

        steps = input_shape[s_axis]
        kernel_w, = self.kernel_size
        stride, = self.strides
        if self.output_padding is None:
            out_pad_w = None
        else:
            out_pad_w, = self.output_padding

        # Infer the dynamic output shape:
        out_width = conv_utils.deconv_length(steps,
                                             stride, kernel_w,
                                             self.padding,
                                             out_pad_w,
                                             self.dilation_rate[0])
        #if self.data_format == 'channels_first':
        #    output_shape = (batch_size, self.filters, 1, out_width)
        #    inputs = K.expand_dims(inputs, axis=2) 
        #else:
        output_shape = (batch_size, 1, out_width, self.filters)
        inputs = K.expand_dims(inputs, axis=1) 

        outputs = K.conv2d_transpose(
            inputs,
            K.expand_dims(self.kernel, axis=0),
            output_shape,
            (1, self.strides[0]),
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=(1,self.dilation_rate[0]))

        #if self.data_format == 'channels_first':
        #    outputs = K.squeeze(outputs, axis=2) 
        #else:
        outputs = K.squeeze(outputs, axis=1) 

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        #if self.data_format == 'channels_first':
        #    c_axis, w_axis = 1, 2
        #else:
        c_axis, w_axis = 2, 1

        kernel_w, = self.kernel_size
        stride, = self.strides
        if self.output_padding is None:
            out_pad_w = None
        else:
            out_pad_w, = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[w_axis] = conv_utils.deconv_length(output_shape[w_axis],
                                                        stride,
                                                        kernel_w,
                                                        self.padding,
                                                        out_pad_w,
                                                        self.dilation_rate[0])
        return tuple(output_shape)

    def get_config(self):
        config = super(Conv1DTranspose, self).get_config()
        config['output_padding'] = self.output_padding
        return config


class RevCompConv1DTranspose(Conv1DTranspose):

    def compute_output_shape(self, input_shape):
        orig_out_shape = super(RevCompConv1DTranspose,
                          self).compute_output_shape(input_shape=input_shape)
        return (orig_out_shape[0], orig_out_shape[1], orig_out_shape[2]*2)

    def call(self, inputs):

        revcomp_kernel =\
            K.concatenate([self.kernel,
                           self.kernel[::-1,::-1,::-1]],axis=-2)
        if (self.use_bias):
            revcomp_bias = K.concatenate([self.bias,
                                          self.bias[::-1]], axis=-1)

        input_shape = K.shape(inputs)
        batch_size = input_shape[0]
        s_axis = 1

        steps = input_shape[s_axis]
        kernel_w, = self.kernel_size
        stride, = self.strides
        if self.output_padding is None:
            out_pad_w = None
        else:
            out_pad_w, = self.output_padding

        # Infer the dynamic output shape:
        out_width = conv_utils.deconv_length(steps,
                                             stride, kernel_w,
                                             self.padding,
                                             out_pad_w,
                                             self.dilation_rate[0])
        output_shape = (batch_size, 1, out_width, 2*self.filters)
        inputs = K.expand_dims(inputs, axis=1) 

        outputs = K.conv2d_transpose(
            inputs,
            K.expand_dims(revcomp_kernel, axis=0),
            output_shape,
            (1, self.strides[0]),
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=(1,self.dilation_rate[0]))
        outputs = K.squeeze(outputs, axis=1) 

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                revcomp_bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class RevCompConv1D(Conv1D):

    def compute_output_shape(self, input_shape):
        length = conv_utils.conv_output_length(input_shape[1],
                                               self.kernel_size[0],
                                               padding=self.padding,
                                               stride=self.strides[0],
                                               dilation=self.dilation_rate[0])
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


