

###
#
#
# for testing: rm -rf TEST; python gravnet.py /eos/cms/store/cmst3/group/hgcal/CMG_studies/gvonsem/hgcalsim/ConverterTask/closeby_1.0To100.0_idsmix_dR0.1_n10_rnd1_s1/dev_LayerClusters_prod2/testconv/dataCollection.dc TEST
#
###

import DeepJetCore
from DeepJetCore.training.training_base import training_base
from DeepJetCore.DataCollection import DataCollection
import keras
from keras.models import Model
from keras.layers import  Reshape, Dense,Conv1D, Conv2D, BatchNormalization, Multiply, Concatenate, Dropout #etc
from Layers import Conv2DGlobalExchange
from DeepJetCore.DJCLayers import ScalarMultiply, Clip, SelectFeatures, Print

from tools import plot_pred_during_training, plot_truth_pred_plus_coords_during_training
import tensorflow as tf
import os

from Losses import per_object_rep_att_loss

nbatch=120 #1*7

plots_after_n_batch=1 #1000
use_event=8
learningrate=1e-4 #-4

momentum=0.6

def model(Inputs,feature_dropout=-1.):

    x = Inputs[0] #this is the self.x list from the TrainData data structure
    x_in = BatchNormalization(momentum=momentum)(x)
    
    feat = []
    for i in range(10):
        x = BatchNormalization(momentum=momentum)(x)  
        if False and not i%2:
            x = Conv2DGlobalExchange()(x)  
        if i and not i % 2:
            x = Dropout(0.05)(x)
        x = Concatenate()([x,x_in])
        x = Conv2D(32+i, (3+2*i,3+2*i), padding='same', activation='elu')(x)
        #x_b = Conv2D(4, (24,24), padding='same', activation='elu')(x)
        #x = Concatenate()([x_a,x_b])
        #feat.append(Conv2D(16, (1,1), padding='same', activation='elu')(x))
    
    #feat=[x]
    #x = Dropout(0.05)(x)
    #x = Concatenate()(feat)
    
    x = Dense(64, activation='elu')(x)
    
    '''
    p_beta   =  tf.reshape(pred[:,:,:,0:1], [pred.shape[0],pred.shape[1]*pred.shape[2],-1])
    p_tpos   =  tf.reshape(pred[:,:,:,1:3], [pred.shape[0],pred.shape[1]*pred.shape[2],-1])
    p_ID     =  tf.reshape(pred[:,:,:,3:6], [pred.shape[0],pred.shape[1]*pred.shape[2],-1])
    p_dim    =  tf.reshape(pred[:,:,:,6:8], [pred.shape[0],pred.shape[1]*pred.shape[2],-1])
    p_object  = pred[:,0,0,8]
    p_ccoords = tf.reshape(pred[:,:,:,8:10], [pred.shape[0],pred.shape[1]*pred.shape[2],-1])
                 
    '''
    
        
    p_beta    = Conv2D(1, (1,1), padding='same',activation='sigmoid',
                       #kernel_initializer='zeros',
                       trainable=True)(x)
    p_tpos    = Conv2D(2, (1,1), padding='same')(x)
    p_ID      = Conv2D(3, (1,1), padding='same',activation='softmax')(x)
    p_dim     = Conv2D(2, (1,1), padding='same')(x)
    #p_object  = Conv2D(1, (1,1), padding='same')(x)
    p_ccoords = Conv2D(2, (1,1), padding='same')(x)
    
    predictions=Concatenate()([p_beta ,  
                               p_tpos   ,
                               p_ID     ,
                               p_dim    ,
                               #p_object ,
                               p_ccoords])
    
    print('predictions',predictions.shape)
    
    return Model(inputs=Inputs, outputs=predictions)



train=training_base(testrun=False,resumeSilently=True,renewtokens=True)

import os
os.system('cp /afs/cern.ch/work/j/jkiesele/HGCal/SOR/modules/betaLosses.py '+train.outputDir+'/')

from tools import plot_pixel_2D_clustering_flat_during_training as plot_pixel_2D_clustering_during_training

#samplepath = "/data/hgcal-0/store/jkiesele/SOR/Dataset/test_wiggle/100.djctd"
samplepath = "/data/hgcal-0/store/jkiesele/SOR/Dataset/test/1.djctd"


def decay_function(ncalls):
    print('call decay')
    if ncalls > 10000:
        return 500
    if ncalls > 1000:
        return 50
    if ncalls > 100:
        return 3
    return 1


ppdts= [plot_pixel_2D_clustering_during_training(
               samplefile=samplepath,
               output_file=train.outputDir+'/train_progress'+str(i),
               use_event=use_event+i,
               afternbatches=plots_after_n_batch,
               on_epoch_end=False,
               mask=False,
               decay_function=decay_function
               ) for i in range(10) ]

from Losses import object_condensation_loss


if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    #for regression use the regression model
    train.setModel(model)
    
    #read weights where possible from pretrained model
    #import os
    #from DeepJetCore.modeltools import load_model, apply_weights_where_possible
    #m_weights =load_model(os.environ['DEEPJETCORE_SUBPACKAGE'] + '/pretrained/gravnet_1.h5')
    #train.keras_model = apply_weights_where_possible(train.keras_model, m_weights)
    
    #for regression use a different loss, e.g. mean_squared_error
train.compileModel(learningrate=learningrate,
                   loss=object_condensation_loss,
                   #clipnorm=1
                   )#metrics=[pixel_over_threshold_accuracy]) 
                  
print(train.keras_model.summary())


ppdts_callbacks=[ppdts[i].callback for i in range(len(ppdts))]

verbosity=2

model,history = train.trainModel(nepochs=1, 
                                 batchsize=int(nbatch),
                                 checkperiod=10, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)


train.change_learning_rate(learningrate/10.)

model,history = train.trainModel(nepochs=1+5, 
                                 batchsize=nbatch,
                                 checkperiod=10, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)

train.change_learning_rate(learningrate/10.)

model,history = train.trainModel(nepochs=1+5+10, 
                                 batchsize=nbatch,
                                 checkperiod=10, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)


model,history = train.trainModel(nepochs=50+200, 
                                 batchsize=nbatch,
                                 checkperiod=50, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)


train.change_learning_rate(learningrate/100.)
model,history = train.trainModel(nepochs=250+250, 
                                 batchsize=nbatch,
                                 checkperiod=50, # saves a checkpoint model every N epochs
                                 verbose=verbosity,
                                 additional_callbacks=ppdts_callbacks)






