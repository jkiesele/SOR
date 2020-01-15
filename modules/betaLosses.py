

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

def calulate_beta_scaling(d,minimum_confidence):
    return (1./(( 1. - d['p_beta'])+K.epsilon()) - 1. + minimum_confidence)

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
    truth = tf.Print(truth,[tf.shape(truth),tf.shape(pred)],'truth, pred ',summarize=30)
    def resh(lastdim):
        return (tf.shape(pred)[0],tf.shape(pred)[1]*tf.shape(pred)[2],lastdim)
    #make it all lists
    outdict['t_mask'] =  tf.reshape(truth[:,:,:,0:1], resh(1)) 
    outdict['t_pos']  =  tf.reshape(truth[:,:,:,1:3], resh(2), name="lala")/16.
    outdict['t_ID']   =  tf.reshape(truth[:,:,:,3:6], resh(3))  
    outdict['t_dim']  =  tf.reshape(truth[:,:,:,6:8], resh(2))/4.
    outdict['t_objidx']= tf.reshape(truth[:,:,:,8:9], resh(1))

    print('pred',pred.shape)

    outdict['p_beta']   =  tf.reshape(pred[:,:,:,0:1], resh(1))
    outdict['p_pos']    =  tf.reshape(pred[:,:,:,1:3], resh(2), name="lulu")/16.
    outdict['p_ID']     =  tf.reshape(pred[:,:,:,3:6], resh(3))
    outdict['p_dim']    =  tf.reshape(pred[:,:,:,6:8], resh(2))/4.
    
    outdict['p_ccoords'] = tf.reshape(pred[:,:,:,8:10], resh(2))
    
    flattened = tf.reshape(outdict['t_mask'],(tf.shape(outdict['t_mask'])[0],-1))
    outdict['n_nonoise'] = tf.expand_dims(tf.cast(tf.count_nonzero(flattened, axis=-1), dtype='float32'), axis=1)
    #will have a meaning for non toy model
    outdict['n_active'] = tf.zeros_like(outdict['n_nonoise'])+64.*64.
    
    
    return outdict

def create_kernel_loss_dict(truth, pred):
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
    
    #make it all lists
    outdict['t_mask'] =  truth[:,:,:,0:1]
    outdict['t_pos']  =  truth[:,:,:,1:3]/16.
    outdict['t_ID']   =  truth[:,:,:,3:6]
    outdict['t_dim']  =  truth[:,:,:,6:8]/4.
    outdict['t_objidx']  = truth[:,:,:,8:9]

    print('pred',pred.shape)

    outdict['p_beta']   =  pred[:,:,:,0:1]
    outdict['p_pos']    =  pred[:,:,:,1:3]/16.
    outdict['p_ID']     =  pred[:,:,:,3:6]
    outdict['p_dim']    =  pred[:,:,:,6:8]/4.
    #p_object  = pred[:,0,0,8]
    outdict['p_ccoords'] = pred[:,:,:,9:]
    
    flattened = tf.reshape(outdict['t_mask'],(tf.shape(outdict['t_mask'])[0],-1))
    outdict['n_nonoise'] = tf.expand_dims(tf.cast(tf.count_nonzero(flattened, axis=-1), dtype='float32'), axis=1)
    #will have a meaning for non toy model
    outdict['n_active'] = tf.zeros_like(outdict['n_nonoise'])+64.*64.
    
    
    return outdict







def make_kernel_indices(truth, kernel_size):
    from lossKernel import makeKernelSelection
    static_kernel = tf.constant(makeKernelSelection(64, kernel_size),dtype='int32') # P*P x 2 needs to go to B * P * P x 3
    
    n_batches = tf.shape(truth)[0]
    
    batch_range = tf.range(0, n_batches)
    batch_range = tf.expand_dims(batch_range, axis=1) # B x 1
    batch_range = tf.expand_dims(batch_range, axis=1) # B x 1 x 1
    batch_range = tf.expand_dims(batch_range, axis=1) # B x 1 x 1 x 1
    
    batch_kernel = tf.expand_dims(static_kernel, axis=0) # 1 x P*P x 2
    batch_kernel = tf.tile(batch_kernel,[n_batches,1,1,1]) # B x P*P x 2
    
    batch_indices = tf.tile(batch_range, [1, tf.shape(batch_kernel)[1], tf.shape(batch_kernel)[2], 1 ]) # B x P*P x 1
    
    indices = tf.concat([batch_indices, batch_kernel], axis=-1)

    return indices


def kernel_euclidean_squared(a,b):
    # B x V x N x F
    #output: B x V x N x 1
    a = a[:,:,0:1,:] # B x V x 1 x F
    return tf.expand_dims(tf.reduce_sum((a-b)**2 , axis=-1), axis=3)


def potential_kernel(truth,pred,minimum_confidence, push_noise=False):
    
    kernel_size = 7
    
    indices = make_kernel_indices(truth, kernel_size)
    
    t_kernel = tf.gather_nd(truth,indices)
    p_kernel = tf.gather_nd(pred,indices)
    
    d = create_kernel_loss_dict(t_kernel,p_kernel)
    
    beta_scaling = 1./(( 1. - d['p_beta'])+K.epsilon()) - 1. + minimum_confidence

    #beta_scaling = tf.Print(beta_scaling,[ tf.shape(beta_scaling)],'beta_scaling ',summarize=100)
    
    dist = tf.sqrt(kernel_euclidean_squared(d['p_ccoords'], d['p_ccoords'])+K.epsilon())
    S    = kernel_euclidean_squared(d['t_objidx'], d['t_objidx'])
    S    = tf.where(S<0.01, tf.zeros_like(S), tf.zeros_like(S)+1.)
    Snot = 1. - S
    
    #dist = tf.Print(dist,[ tf.shape(dist)],'dist ',summarize=100)
    
    N_nonoise = tf.cast(tf.count_nonzero(d['t_mask'],axis=2), dtype='float32')
    
    #N_nonoise = tf.Print(N_nonoise,[ N_nonoise],'N_nonoise ',summarize=2000)
    
    S *= d['t_mask']
    if not push_noise:
        Snot *= d['t_mask']
    
    #p_beta_kernel  = tf.expand_dims(d['p_beta'], axis = 2) # B x V X 1 x N x 1
    #p_beta_kernel *= tf.expand_dims(d['p_beta'], axis = 3) # B x V X N x N x 1
      
    r_dist = 1. - dist
    r_dist = tf.where(r_dist < 0, tf.zeros_like(r_dist),r_dist)
    
    repulsion =  d['t_mask'] * beta_scaling * Snot * r_dist
    if not push_noise:
        repulsion = tf.reduce_sum(repulsion, axis=2) / (N_nonoise + K.epsilon())
    else:
        repulsion = tf.reduce_sum(repulsion, axis=2) / float(kernel_size)
    
    attraction = d['t_mask'] * beta_scaling * S * dist
    attraction = tf.reduce_sum(attraction, axis=2)/ (N_nonoise + K.epsilon())
    
    return repulsion, attraction

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

def min_beta_pen(d, return_objmask=False):
    max_beta=[]
    mask_obj=[]
    asso_beta = tf.where(d['t_mask']>0.5, d['p_beta'], tf.zeros_like(d['p_beta']))
    #for loop for minbeta part
    for i in range(9):#maximum number of objects
        beta_i = tf.where(tf.abs(d['t_objidx']-float(i))<0.2, asso_beta, tf.zeros_like(asso_beta))
        maxbeta_i = tf.reduce_max(beta_i, axis=1) 
        max_beta.append(maxbeta_i)
        mask_i = tf.where(tf.abs(d['t_objidx']-float(i))<0.2, tf.zeros_like(d['p_beta'])+1, tf.zeros_like(d['p_beta']))
        mask_i = tf.reduce_max(mask_i, axis=1) 
        mask_obj.append(mask_i)
        
        
    max_beta = tf.concat(max_beta,axis=1)# B x 9
    isobj = tf.concat(mask_obj,axis=1)# B x 9
    
    max_beta = tf.Print(max_beta,[max_beta[0], isobj[0], tf.shape(max_beta), tf.shape(isobj)],'max_beta, mask_obj ',summarize=30)
    #max_beta = tf.Print(max_beta,[max_beta],'max_beta ', summarize=70)
    
    N_obj = tf.cast(tf.count_nonzero(isobj, axis=1),dtype='float32')
    
    #N_obj = tf.Print(N_obj,[N_obj],'N_obj ', summarize=70)
    
    mean_max_beta_pen = tf.reduce_sum( isobj*(1. - max_beta), axis=1) / (N_obj + K.epsilon()) #B x 1
    if return_objmask:
        return tf.reduce_mean(mean_max_beta_pen), isobj, N_obj
    return tf.reduce_mean(mean_max_beta_pen)

def cross_entr_loss(d, beta_scaling):
    tID = d['t_mask']*d['t_ID']
    tID = tf.where(tID<=0.,tf.zeros_like(tID)+10*K.epsilon(),tID)
    tID = tf.where(tID>=1.,tf.zeros_like(tID)+1.-10*K.epsilon(),tID)
    pID = d['t_mask']*d['p_ID']
    pID = tf.where(pID<=0.,tf.zeros_like(pID)+10*K.epsilon(),pID)
    pID = tf.where(pID>=1.,tf.zeros_like(pID)+1.-10*K.epsilon(),pID)
    
    xentr = beta_scaling * (-1.)* tf.reduce_sum(tID * tf.log(pID) ,axis=-1, keepdims=True)
    xentr_loss = mean_nvert_with_nactive(d['t_mask']*xentr, d['n_nonoise'])
    #xentr_loss = tf.where(tf.is_nan(xentr_loss), tf.zeros_like(xentr_loss)+10., xentr_loss)
    return tf.reduce_mean(xentr_loss)

def pos_loss(d, beta_scaling):
    posl = beta_scaling*tf.abs(d['t_pos'] - d['p_pos'])
    posl = mean_nvert_with_nactive(d['t_mask']*posl,d['n_nonoise'])
    #posl = tf.where(tf.is_nan(posl), tf.zeros_like(posl)+10., posl)
    return tf.reduce_mean( posl)

def box_loss(d, beta_scaling):
    bboxl = beta_scaling*tf.abs(d['t_dim'] - d['p_dim'])
    bboxl = mean_nvert_with_nactive(d['t_mask']*bboxl,d['n_nonoise'])
    #bboxl = tf.where(tf.is_nan(bboxl), tf.zeros_like(bboxl)+10., bboxl)
    return tf.reduce_mean( bboxl)
    
def sup_noise_loss(d):
    return tf.reduce_mean(mean_nvert_with_nactive(((1.-d['t_mask'])*d['p_beta']), 
                                            tf.abs(d['n_active']-d['n_nonoise']))  )  

def per_object_rep_att_loss(truth,pred):
    
    d = create_pixel_loss_dict(truth,pred)
    beta_scaling = calulate_beta_scaling(d,minimum_confidence=1e-2)
    betaloss, isobj_in, N_obj_in = min_beta_pen(d, return_objmask=True)
    
    att = []
    rep = []
    nrep=[]
    isobj =[]
    
    
    maxobjs=9
    
    
    N_obj = tf.where(N_obj_in>maxobjs, tf.zeros_like(N_obj_in)+maxobjs, N_obj_in)
    
    
    randidx = tf.random.shuffle(tf.constant(range(9),dtype='int32'))
    # in principle we can even put this on a different GPU
    for iii in range(maxobjs):#maximum number of objects / but this even works for more as long as the index is randomised!
        i = randidx[iii]
        
        isobj.append(isobj_in[:,i:i+1])#keep the last dimension for concat later
        
        S_i     = d['t_mask']*tf.where(tf.abs(d['t_objidx']-tf.cast(i,dtype='float32'))<0.1, tf.zeros_like(d['t_objidx'])+1, tf.zeros_like(d['t_objidx']))
        
        # the distinction between noise and other objects is only sensible if there is a clear imbalance of objects and noise
        #
        S_not_i = d['t_mask']*(1. - S_i)
        S_noise_i = (1 - d['t_mask'])*(1. - S_i)
        
        #make something like S_not_noise, and S_not, and normalise differently
        #build the mean
        Nw_i    = tf.reduce_sum(beta_scaling * S_i, axis=1, keepdims=True) # B x 1 x 1 
        N_i     = tf.reduce_sum(S_i, axis=1)    # B x 1
        Nnot_i  = tf.reduce_sum(S_not_i, axis=1) # B x 1 
        Nnoise_i  = tf.reduce_sum(S_noise_i, axis=1) # B x 1 
        pos_i   = tf.reduce_sum(beta_scaling * S_i * d['p_ccoords'], axis=1, keepdims=True)/(Nw_i+K.epsilon()) #B x 1 x 2
        
        #S_i: B x V x 1, pos_i: B x 1 x 1
        distance = tf.sqrt(tf.reduce_sum((pos_i - d['p_ccoords'])**2, axis=2, keepdims=True) + K.epsilon()) #B x V x 1
        attraction = beta_scaling * S_i * distance
        
        repulsion  = beta_scaling * S_not_i * (1 - distance)
        repulsion  = tf.where(repulsion < 0, tf.zeros_like(repulsion),repulsion)
        
        noise_repulsion = beta_scaling * S_noise_i * (1 - distance)
        noise_repulsion = tf.where(noise_repulsion < 0, tf.zeros_like(noise_repulsion),noise_repulsion)
        #so far everything B x V x 1
        attraction = tf.reduce_sum(attraction, axis=1)/(N_i + K.epsilon())
        repulsion = tf.reduce_sum(repulsion, axis=1)/(Nnot_i + K.epsilon())
        noise_repulsion = tf.reduce_sum(noise_repulsion, axis=1)/(Nnoise_i + K.epsilon())
        
        repulsion += noise_repulsion
        #now B x 1
        att.append(attraction)
        rep.append(repulsion)
        
        

    print('N_obj',N_obj.shape)
    isobj = tf.concat(isobj,axis=1)
    att = tf.concat(att,axis=1)
    rep = tf.concat(rep,axis=1)
    
    att = tf.Print(att,[att, tf.shape(att), N_obj],'att, N_obj ')
    
    #
    # Do we actually want to normalise this with the total number of objects? maybe we want more 
    # weight for a multi-object image, because it is also harder to learn?
    #
    attraction = tf.reduce_sum(isobj * att, axis=1) # B  
    attraction /= N_obj + K.epsilon()
    attraction = tf.Print(attraction,[tf.shape(attraction)],'attraction ')
    attloss = tf.reduce_mean(attraction)

    repulsion  = tf.reduce_sum(isobj * rep, axis=1) # B x 9  
    repulsion  /= N_obj + K.epsilon()
    reploss = tf.reduce_mean(repulsion)
    
    #sets in at beta = 0.5
    beta_scaling = tf.where(beta_scaling>1, beta_scaling-1., tf.zeros_like(beta_scaling))
    
    xentr_loss = cross_entr_loss(d, beta_scaling)
    posl = pos_loss(d, beta_scaling)
    bboxl = box_loss(d, beta_scaling)
    supress_noise_loss = sup_noise_loss(d)
    
    loss = reploss + attloss + betaloss + supress_noise_loss  + posl + bboxl + xentr_loss
    
    loss = tf.Print(loss,[loss,
                              reploss,
                              attloss,
                              betaloss,
                              supress_noise_loss,
                              posl,
                              bboxl,
                              xentr_loss
                              ],
                              'loss, repulsion_loss, attraction_loss, min_beta_loss, supress_noise_loss, pos_loss, bboxl, xentr_loss  ' )
    
    return loss


def kernel_loss(truth,pred):
    
    #
    #
    # Bound to fail. Make the whole loss a loop over the objects and distances to means
    # The mean calculation must include beta as weight to be consistent
    # This is exact for the attractive potential, and an approximation for the repulsive one
    #
    #
    
    
    minimum_confidence = 1e-3
    
    repulsion, attraction = potential_kernel(truth,pred,minimum_confidence,push_noise=False)
    
    d = create_pixel_loss_dict(truth, pred)
    beta_scaling = 1./(( 1. - d['p_beta'])+K.epsilon()) - 1. + minimum_confidence #B x V x 1
    
    repulsion = tf.reduce_sum(d['t_mask']*beta_scaling *repulsion, axis=1)/ (d['n_nonoise']+K.epsilon()) #B x 1
    attraction = tf.reduce_sum(d['t_mask']*beta_scaling*attraction, axis=1)/ (d['n_nonoise']+K.epsilon()) #B x 1
    
    mean_max_beta_pen = min_beta_pen(d)
    
    reploss, attloss, betaloss = tf.reduce_mean(repulsion), tf.reduce_mean(attraction), tf.reduce_mean(mean_max_beta_pen)
    
    
    ID_strength =  1.
    pos_strength = 1.
    box_strength = 1.
    
    
    
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

    posl = mean_nvert_with_nactive(d['t_mask']*posl,d['n_nonoise'])
    #posl = tf.where(tf.is_nan(posl), tf.zeros_like(posl)+10., posl)
    posl = tf.reduce_mean( posl)
    
    ######### bounding box loss
    
    
    bboxl = box_strength * beta_scaling*tf.abs(d['t_dim'] - d['p_dim'])
    bboxl = mean_nvert_with_nactive(d['t_mask']*bboxl,d['n_nonoise'])
    #bboxl = tf.where(tf.is_nan(bboxl), tf.zeros_like(bboxl)+10., bboxl)
    bboxl = tf.reduce_mean( bboxl)
    
    
    #######################################################
    
    
    supress_noise_loss = mean_nvert_with_nactive(((1.-d['t_mask'])*d['p_beta']), 
                                            tf.abs(d['n_active']-d['n_nonoise']))
    
    betaloss = tf.reduce_mean(betaloss)
    supress_noise_loss = tf.reduce_mean(supress_noise_loss)
    bboxl = tf.reduce_mean(bboxl)
    xentr_loss = tf.reduce_mean(xentr_loss)
    posl = tf.reduce_mean(posl)
    reploss = tf.reduce_mean(reploss)
    attloss = tf.reduce_mean(attloss)
    
    loss = betaloss + supress_noise_loss + bboxl  + xentr_loss + posl # 
    loss += reploss + attloss 
    
    loss = tf.Print(loss,[loss,
                              reploss,
                              attloss,
                              betaloss,
                              supress_noise_loss,
                              posl,
                              bboxl,
                              xentr_loss
                              ],
                              'loss, repulsion_loss, attraction_loss, min_beta_loss, supress_noise_loss, pos_loss, bboxl, xentr_loss  ' )
    return loss


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
    
    #new implementation with explicit object loop
    min_beta_loss = min_beta_pen(d)
    
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
    pass

