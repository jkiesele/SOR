


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


class PadTracker(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(PadTracker, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],2*32,2*32,input_shape[3])
    
    def call(self, inputs):
        return tf.pad(inputs, [[0,0],[16,16],[16,16],[0,0]])
        
    
    def get_config(self):
        base_config = super(PadTracker, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['PadTracker']=PadTracker 


class CropTracker(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(CropTracker, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],32,32,input_shape[3])
    
    def call(self, inputs):
        return inputs[:,16:48,16:48,:]
        
    
    def get_config(self):
        base_config = super(CropTracker, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['CropTracker']=CropTracker 


class TileTrackerFeatures(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(TileTrackerFeatures, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]*4)
    
    def call(self, inputs):
        return tf.tile(inputs, [1,1,1,4])
        
    
    def get_config(self):
        base_config = super(TileTrackerFeatures, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['TileTrackerFeatures']=TileTrackerFeatures 



class TileCalo(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(TileCalo, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],4*16,4*16,input_shape[3])
    
    def call(self, inputs):
        return tf.tile(inputs, [1,4,4,1])
        
    
    def get_config(self):
        base_config = super(TileCalo, self).get_config()
        return dict(list(base_config.items()))
        

global_layers_list['TileCalo']=TileCalo 

class Tile2D(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, ntiles, **kwargs):
        super(Tile2D, self).__init__(**kwargs)
        self.ntiles=ntiles
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],4*16,4*16,input_shape[3])
    
    def call(self, inputs):
        return tf.tile(inputs, [1,self.ntiles,self.ntiles,1])
        
    
    def get_config(self):
        base_config = super(Tile2D, self).get_config()
        config = {'ntiles' :self.ntiles}
        return dict(list(config.items() ) + list(base_config.items()) )
        

global_layers_list['Tile2D']=Tile2D 


class GaussActivation(Layer):
    '''
    Centers phi to the first input vertex, such that the 2pi modulo behaviour 
    disappears for a small selection
    '''
    def __init__(self, **kwargs):
        super(GaussActivation, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def call(self, inputs):
        return tf.exp(- inputs**2 )
        
    
    def get_config(self):
        base_config = super(GaussActivation, self).get_config()
        return dict(list(base_config.items()) )
        

global_layers_list['GaussActivation']=GaussActivation 









