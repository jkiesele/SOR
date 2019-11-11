


from keras.layers import Layer
import keras.backend as K
import tensorflow as tf

# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}

class Conv2DGlobalExchange(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(Conv2DGlobalExchange, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]+input_shape[3])
    
    def call(self, inputs):
        average = tf.reduce_mean(inputs, axis=[1,2], keepdims=True)
        average = tf.tile(average, [1,tf.shape(inputs)[1],tf.shape(inputs)[2],1])
        return tf.concat([inputs,average],axis=-1)
        
    
    def get_config(self):
        base_config = super(Conv2DGlobalExchange, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['Conv2DGlobalExchange']=Conv2DGlobalExchange 