from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.engine import Layer, InputSpec
from keras.layers.normalization import BatchNormalization
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K


class RevCompConv1DBatchNorm(BatchNormalization):
    '''Batch norm that shares weights over reverse complement channels
    '''

    def build(self, input_shape):
        self.num_input_chan = input_shape[self.axis]
        self.input_len = input_shape[1]
        modified_input_shape = [input_shape[0],
                                input_shape[1]*2,
                                int(input_shape[2]/2)]
        super(RevCompConv1DBatchNorm, self).build(modified_input_shape)
        self.input_spec = InputSpec(
                            ndim=len(input_shape),
                            axes={self.axis: self.num_input_chan})

    def call(self, inputs, training=None):
        orig_inputs = inputs
        #create a fake input by concatentating reverse-complemented pairs
        #along the length dimension
        inputs = K.concatenate(
            tensors=[inputs[:,:,:int(self.num_input_chan/2)],
                     inputs[:,:,int(self.num_input_chan/2):][:,:,::-1]],
            axis=1)

        input_shape = K.int_shape(inputs)
        #Prepare broadcasting shape
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1]*len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        #Determines when broadcasting is needed
        #I guess broadcasting is needed when the broadcasting axes are
        #not just a list of the first few axes before the first
        #non-broadcast dimension
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if (needs_broadcasting):
                #In this case we must explicitly broadcast all parameters 
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if (self.center):
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if (self.scale):
                    broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                        inputs,
                        broadcast_moving_mean,
                        broadcast_moving_variance,
                        broadcast_beta,
                        broadcast_gamma,
                        #axis=self.axis,
                        epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                        inputs,
                        self.moving_mean,
                        self.moving_variance,
                        self.beta,
                        self.gamma,
                        #axis=self.axis,
                        epsilon=self.epsilon) 

        #If the learning phase is *static* and set to inference: 
        if training in {0, False}:
            normed_inputs = normalize_inference()
        else:
            #If the learning is either dynamic or set to training:
            normed_training, mean, variance = K.normalize_batch_in_training(
                inputs, self.gamma, self.beta, reduction_axes,
                epsilon=self.epsilon) 
            
            if K.backend() != 'cntk':
                sample_size = K.prod([K.shape(inputs)[axis]
                                      for axis in reduction_axes]) 
                sample_size = K.cast(sample_size, dtype=K.dtype(inputs))
                # sample vairance - unbiased estimator of population variance
                variance *= sample_size / (sample_size - (1.0 + self.epsilon))

            self.add_update([K.moving_average_update(self.moving_mean,
                                                     mean,
                                                     self.momentum),
                             K.moving_average_update(self.moving_variance,
                                                     variance,
                                                     self.momentum)],
                            inputs)
            normed_inputs = K.in_train_phase(normed_training,
                                             normalize_inference,
                                             training=training)
        
        true_normed_inputs = K.concatenate(
            tensors=[normed_inputs[:,:self.input_len,:],
                     normed_inputs[:,self.input_len:,:][:,:,::-1]],
            axis=2)
        return true_normed_inputs
