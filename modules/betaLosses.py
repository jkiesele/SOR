

import tensorflow as tf
import keras
from keras import losses
import keras.backend as K

#factorise a bit


#
#
#
#  all inputs/outputs of dimension B x V x F (with F being 1 in some cases)
#
#

def create_pixel_loss_dict(truth, pred):
    '''
    input features as
    B x P x P x F
    with F = colours
    
    truth as 
    B x P x P x T
    with T = [mask, true_posx, true_posy, ID_0, ID_1, ID_2, true_width, true_height, n_objects]
    
    all outputs in B x V x 1/F form except
    n_active: B x 1
    
    '''
    outdict={}
    #truth = tf.Print(truth,[truth],'truth',summarize=30)
    reshaping = (tf.shape(pred)[0],tf.shape(pred)[1]*tf.shape(pred)[2],-1)
    #make it all lists
    outdict['t_mask'] =  tf.reshape(truth[:,:,:,0:1], reshaping) 
    outdict['t_pos']  =  tf.reshape(truth[:,:,:,1:3], reshaping, name="lala")/16.
    outdict['t_ID']   =  tf.reshape(truth[:,:,:,3:6], reshaping)  
    outdict['t_dim']  =  tf.reshape(truth[:,:,:,6:8], reshaping)/4.
    n_objects = truth[:,0,0,8]

    print('pred',pred.shape)

    outdict['p_beta']   =  tf.reshape(pred[:,:,:,0:1], reshaping)
    outdict['p_pos']    =  tf.reshape(pred[:,:,:,1:3], reshaping, name="lulu")/16.
    outdict['p_ID']     =  tf.reshape(pred[:,:,:,3:6], reshaping)
    outdict['p_dim']    =  tf.reshape(pred[:,:,:,6:8], reshaping)/4.
    p_object  = pred[:,0,0,8]
    outdict['p_ccoords'] = tf.reshape(pred[:,:,:,9:], reshaping)
    
    flattened = tf.reshape(outdict['t_mask'],(tf.shape(outdict['t_mask'])[0],-1))
    outdict['n_nonoise'] = tf.expand_dims(tf.cast(tf.count_nonzero(flattened, axis=-1), dtype='float32'), axis=1)
    #will have a meaning for non toy model
    outdict['n_active'] = tf.zeros_like(outdict['n_nonoise'])+64.*64.
    
    
    return outdict


def d_euclidean_squared(a,b):
    # B x V x F
    #output: B x V x V x 1
    a = tf.expand_dims(a, axis=1) # B x 1 x V x F
    b = tf.expand_dims(b, axis=2) # B x V x 1 x F
    return tf.expand_dims(tf.reduce_sum((a-b)**2 , axis=-1), axis=3)


def expand_by_mult(A):
    '''
    A: B x V x 1
    out: B x V x V x 1, with out_ij = Ai * Aj
    '''
    assert len(A.shape) == 3
    A1 = tf.expand_dims(A, axis=1) # B x 1 X V x 1
    A2 = tf.expand_dims(A, axis=2) # B x V X 1 x 1
    A =  A1 * A2 # B x V X V x 1
    return A

def mean_nvert_with_nactive(A, n_active):
    '''
    n_active: B x 1
    A : B x V x F
    
    out: B x F
    '''
    assert len(A.shape) == 3
    assert len(n_active.shape) == 2
    den = n_active + K.epsilon()
    ssum = tf.reduce_sum(A, axis=1)
    return ssum / den


def matrix_mean_nvert_with_nactive(AM, n_active):
    '''
    n_active: B x 1
    AM : B x V x V x F
    
    out: B x V x F # mean on second V dimension
    '''
    assert len(AM.shape) == 4
    assert len(n_active.shape) == 2
    den = tf.expand_dims(n_active, axis=1) + K.epsilon()
    ssum = tf.reduce_sum(AM, axis=2)
    return ssum/den
    
def create_others_and_self_masks(true_pos):
    true_dist = d_euclidean_squared(true_pos, true_pos)
    o = tf.where(true_dist>K.epsilon(),tf.zeros_like(true_dist)+1., tf.zeros_like(true_dist))
    s = tf.where(o>K.epsilon(),tf.zeros_like(o), tf.zeros_like(o)+1.)
    return s, o
    
    
def coord_loss(truth,pred, pretrain=False, alllinear=False):
    
    minimum_confidence = 1e-2
    distance_scale = 1.
    repulse_exp = False
    repulse_lin = False
    attract_lin = False
    if alllinear:
        repulse_lin = True
        attract_lin = True
    repulse_scale = 1.
    repulse_y_max = 1. #only for non exp mode
    repulsion_strength = 1.
    attraction_strength = 1.
    supress_noise_strength = 1.
    beta_strength = 1.
    
    ID_strength =  1.
    pos_strength = 1.
    box_strength = 1.
    
    d = create_pixel_loss_dict(truth,pred)
    
    ### zero beta for noise
    supress_noise_loss = mean_nvert_with_nactive(((1.-d['t_mask'])*d['p_beta']), 
                                            tf.abs(d['n_active']-d['n_nonoise']))
    #supress_noise_loss = tf.where(tf.is_nan(supress_noise_loss), 
    #                              tf.zeros_like(supress_noise_loss)+10., supress_noise_loss)
    #B x 1
    supress_noise_loss = supress_noise_strength * tf.reduce_mean(supress_noise_loss)

    ### beta scaling multiplicative
    beta_scaling = 1./(( 1. - d['p_beta'])+K.epsilon()) - 1. + minimum_confidence #B x V x 1
    
    print('beta_scaling', beta_scaling.shape)
    beta_scaling_2D = expand_by_mult(beta_scaling)
    
    t_mask_2D = expand_by_mult(d['t_mask'])
    print('t_mask_2D', t_mask_2D.shape)

    distance = d_euclidean_squared(d['p_ccoords'], d['p_ccoords'])
    
    self_mask, other_mask = create_others_and_self_masks(d['t_pos'])
    print('self_mask', self_mask.shape)
    print('other_mask', other_mask.shape)
    print('t_mask_2D',t_mask_2D.shape)
    print('beta_scaling_2D',beta_scaling_2D.shape)

    attraction_loss = 0
    repulsion_loss = 0
    if not pretrain:
        #repulsion
        repulsion=None
        if repulse_exp:
            repulsion =  t_mask_2D * beta_scaling_2D * other_mask * tf.exp(-0.5/repulse_scale*tf.sqrt(distance+K.epsilon())) 
        else:
            repulsion =  t_mask_2D * beta_scaling_2D * other_mask * 1./(distance/repulse_scale**2 + 1/repulse_y_max)
        if repulse_lin:
            r_dist = 1 - tf.sqrt(distance+K.epsilon())
            r_dist = tf.where(r_dist < 0, tf.zeros_like(r_dist),r_dist)
            repulsion =  t_mask_2D * beta_scaling_2D * other_mask * r_dist
            
        repulsion = matrix_mean_nvert_with_nactive(repulsion, d['n_nonoise'])
        print('repulsion',repulsion.shape)
        repulsion_loss = repulsion_strength * mean_nvert_with_nactive(repulsion, d['n_nonoise'])
        print('repulsion_loss',repulsion_loss.shape)
        # B x 1
        #repulsion_loss = tf.where(tf.is_nan(repulsion_loss), tf.zeros_like(repulsion_loss)+10., repulsion_loss)
        repulsion_loss = tf.reduce_mean(repulsion_loss)
        
        attraction = t_mask_2D * beta_scaling_2D * self_mask * 1./distance_scale**2 * distance
        if attract_lin:
            attraction = t_mask_2D * beta_scaling_2D * self_mask * tf.sqrt(distance+K.epsilon())
        attraction = matrix_mean_nvert_with_nactive(attraction, d['n_nonoise'])
        attraction_loss = attraction_strength * mean_nvert_with_nactive(attraction, d['n_nonoise'])
        # B x 1
        #attraction_loss = tf.where(tf.is_nan(attraction_loss), tf.zeros_like(attraction_loss)+10., attraction_loss)
        attraction_loss = tf.reduce_mean(attraction_loss)
    
    
    ######### beta penalty, one per. #B x V x 1
    self_invbeta = tf.tile(tf.expand_dims( 1. - d['p_beta'] , axis=1), [1,tf.shape(d['p_beta'])[1] ,1, 1] )
    #B x V x V x 1
    print('self_invbeta',self_invbeta.shape)
    
    #othat mask includes noise! so that's like 1-S 
    #other_mask= tf.Print(other_mask,[other_mask],'other_mask ', summarize=200)
    self_invbeta += other_mask * (tf.zeros_like(other_mask)+10) #make sure the other parts don't contribute
    self_min_beta = d['t_mask']*tf.reduce_min(self_invbeta, axis=2)
    print('self_min_beta',self_min_beta.shape)
    min_beta_loss = mean_nvert_with_nactive(self_min_beta, d['n_nonoise'])
    print('min_beta_loss',min_beta_loss.shape)
    #B x 1
    #min_beta_loss = tf.where(tf.is_nan(min_beta_loss), tf.zeros_like(min_beta_loss)+10., min_beta_loss)
    min_beta_loss = beta_strength * tf.reduce_mean(min_beta_loss)
    
    
    ########### Actual losses for ID etc. #################
    tID = d['t_mask']*d['t_ID']
    tID = tf.where(tID<=0.,tf.zeros_like(tID)+10*K.epsilon(),tID)
    tID = tf.where(tID>=1.,tf.zeros_like(tID)+1.-10*K.epsilon(),tID)
    pID = d['t_mask']*d['p_ID']
    pID = tf.where(pID<=0.,tf.zeros_like(pID)+10*K.epsilon(),pID)
    pID = tf.where(pID>=1.,tf.zeros_like(pID)+1.-10*K.epsilon(),pID)
    
    xentr = beta_scaling * (-1.)* tf.reduce_sum(tID * tf.log(pID) ,axis=-1, keepdims=True)
    xentr_loss = mean_nvert_with_nactive(d['t_mask']*xentr, d['n_nonoise'])
    #xentr_loss = tf.where(tf.is_nan(xentr_loss), tf.zeros_like(xentr_loss)+10., xentr_loss)
    xentr_loss = ID_strength * tf.reduce_mean(xentr_loss)
   
   
    ######### position loss


    posl = pos_strength  * beta_scaling*tf.abs(d['t_pos'] - d['p_pos'])
    if pretrain:
        posl += pos_strength  * beta_scaling*tf.abs(d['t_pos'] - d['p_ccoords'])
    posl = mean_nvert_with_nactive(d['t_mask']*posl,d['n_nonoise'])
    #posl = tf.where(tf.is_nan(posl), tf.zeros_like(posl)+10., posl)
    posl = tf.reduce_mean( posl)
    
    ######### bounding box loss
    
    
    bboxl = box_strength * beta_scaling*tf.abs(d['t_dim'] - d['p_dim'])
    bboxl = mean_nvert_with_nactive(d['t_mask']*bboxl,d['n_nonoise'])
    #bboxl = tf.where(tf.is_nan(bboxl), tf.zeros_like(bboxl)+10., bboxl)
    bboxl = tf.reduce_mean( bboxl)
    
    
    #######################################################
    
    
    loss = min_beta_loss + supress_noise_loss + bboxl  + xentr_loss + posl # 
    if not pretrain:
        loss += repulsion_loss + attraction_loss 
    loss = tf.debugging.check_numerics(loss, 'loss has numerical errors')
    if not pretrain:
        loss = tf.Print(loss,[loss,
                              repulsion_loss,
                              attraction_loss,
                              min_beta_loss,
                              supress_noise_loss,
                              posl,
                              bboxl,
                              xentr_loss
                              ],
                              'loss, repulsion_loss, attraction_loss, min_beta_loss, supress_noise_loss, pos_loss, bboxl, xentr_loss  ' )
    else:
        loss = tf.Print(loss,[loss,
                              min_beta_loss,
                              supress_noise_loss,
                              posl,
                              bboxl,
                              xentr_loss
                              ],
                              'loss, min_beta_loss, supress_noise_loss, pos_loss, bboxl, xentr_loss  ' )
    
    return loss


def true_euclidean_squared(a,b):
    # B x V x F
    #output: B x V x V 
    a = tf.expand_dims(a, axis=1) # B x 1 x V x F
    b = tf.expand_dims(b, axis=2) # B x V x 1 x F
    return tf.reduce_sum((a-b)**2 , axis=-1)


def get_n_active_pixels(p):
    flattened = tf.reshape(p,(tf.shape(p)[0],-1))
    return tf.cast(tf.count_nonzero(flattened, axis=-1), dtype='float32')


def mean_with_mask(n_vertices,in_vertices, axis, addepsilon=0.):
    import keras.backend as K
    return tf.reduce_sum(in_vertices,axis=axis) / (n_vertices+K.epsilon()+addepsilon)    




def convolutional_kernel_beta_loss(truth,pred):















