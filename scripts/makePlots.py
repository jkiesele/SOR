#!/usr/bin/env python

import numpy as np
from DeepJetCore.TrainData import TrainData
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import math  
from numba import jit

def makedict(pred,feat,truth):
    
    outdict = {}
    outdict['t_mask'] =  truth[:,:,:,0:1]
    outdict['t_pos']  =  truth[:,:,:,1:3]
    outdict['t_ID']   =  truth[:,:,:,3:6]
    outdict['t_dim']  =  truth[:,:,:,6:8]
    n_objects = truth[:,0,0,8]

    outdict['p_beta']   =  pred[:,:,:,0:1]
    outdict['p_pos']    =  pred[:,:,:,1:3]
    outdict['p_ID']     =  pred[:,:,:,3:6]
    outdict['p_dim']    =  pred[:,:,:,6:8]
    
    outdict['p_ccoords'] = pred[:,:,:,9:]
    
    
    outdict['f_rgb'] = feat[:,:,:,0:3]
    outdict['f_xy'] = feat[:,:,:,3:]
    
    return outdict
        
def maskbeta(datadict, dims, threshold):
    betamask = np.tile(datadict['p_beta'], [1,1,1,dims])
    betamask[betamask>threshold] = 1
    betamask[betamask<=threshold] = 0
    return betamask


@jit(nopython=True)        
def c_collectoverthresholds(betas, 
                            ccoords, 
                            sorting,
                            betasel,
                          beta_threshold, distance_threshold):
    

    for e in range(len(betasel)):
        selected = []
        for si in range(len(sorting[e])):
            i = sorting[e][si]
            use=True
            for s in selected:
                distance = math.sqrt( (s[0]-ccoords[e][i][0])**2 +  (s[1]-ccoords[e][i][1])**2 )
                if distance  < distance_threshold:
                    use=False
                    break
            if not use:
                betasel[e][i] = False
                continue
            else:
                selected.append(ccoords[e][i])
             
    return betasel
    
def collectoverthresholds(data, 
                          beta_threshold, distance_threshold):
    
    betas   = np.reshape(data['p_beta'], [data['p_beta'].shape[0], -1])
    ccoords = np.reshape(data['p_ccoords'], [data['p_ccoords'].shape[0], -1, data['p_ccoords'].shape[3]])
    
    sorting = np.argsort(-betas, axis=1)
    
    betasel = betas > beta_threshold
    
    bsel =  c_collectoverthresholds(betas, 
                            ccoords, 
                            sorting,
                            betasel,
                          beta_threshold, distance_threshold)
    
    
    return np.reshape(bsel , [data['p_beta'].shape[0], data['p_beta'].shape[1], data['p_beta'].shape[2]])
    

            
            
            
            
            
       

parser = ArgumentParser('make plots')
parser.add_argument('inputFile')


args = parser.parse_args()

#use traindata as data storage
td = TrainData()
td.readFromFile(args.inputFile)

data = makedict(td.x[0],td.x[1],td.x[2])


betaselection = collectoverthresholds(data, 0.3, 0.8) #0.2/2.0

print('betaselection',betaselection.shape)

def makeRectangle(size, pos,edgecolor='y'):
    return patches.Rectangle([pos[0]-size[0]/2.,pos[1]-size[1]/2.],size[0],size[1],linewidth=1,edgecolor=edgecolor,facecolor='none')

true_recspos = np.reshape(data['t_pos'], [data['t_pos'].shape[0],-1,data['t_pos'].shape[-1]]) 
true_recsdim = np.reshape(data['t_dim'], [data['t_dim'].shape[0],-1,data['t_dim'].shape[-1]])
pred_recpos  = np.reshape(data['p_pos'], [data['p_pos'].shape[0],-1,data['p_pos'].shape[-1]]) 
pred_recsdim = np.reshape(data['p_dim'], [data['p_dim'].shape[0],-1,data['p_dim'].shape[-1]]) 

flat_betaselection = np.reshape(betaselection, [betaselection.shape[0],-1])

print('flat_betaselection',flat_betaselection.shape)

pIDimage = data['p_ID']
pIDimage[np.invert(betaselection)] *=0
    
for i in range(min(len(td.x[0]), 30)):
    
    
    fig,ax = plt.subplots(1,1) #.subplots(2,2)
    ax.set_aspect(aspect=1.)
    
    image = data['f_rgb'][i]/256.
    
    #make everything more damp
    image[image<0.99] /= 1.5
    
    ax.imshow(image, aspect=1)

    these_true_positions = []
    
    for j in range(len(true_recspos[i])):
        if (true_recspos[i][j][0],true_recspos[i][j][1]) in these_true_positions: continue
        these_true_positions.append((true_recspos[i][j][0],true_recspos[i][j][1]))
        
        rec = makeRectangle(true_recsdim[i][j], true_recspos[i][j] ,'r')
        ax.add_patch(rec)
    plt.tight_layout()
    fig.savefig("true_image"+str(i)+".pdf")
    
    plt.close()
    fig,ax =  plt.subplots(1,1)
    ax.imshow(image, aspect=1)
    ax.set_aspect(aspect=1.)
    
    sel_pix = pIDimage[i]
    sel_pix_alpha = np.expand_dims(np.sum(sel_pix, axis=-1), axis=3)
    sel_pix_alpha[sel_pix_alpha>0]=1.
    sel_pix = np.concatenate([sel_pix,sel_pix_alpha],axis=-1)
    ax.imshow(sel_pix, aspect=1)
    
    nrecs = 0
    for j in range(len(flat_betaselection[i])):
        if flat_betaselection[i][j]:
            rec = makeRectangle(pred_recsdim[i][j], pred_recpos[i][j],'y' )
            ax.add_patch(rec)
            nrecs += 1
    print('nrecs ',nrecs)
    
    fig.savefig("pred_image"+str(i)+".pdf")
    plt.close()
    fig,ax =  plt.subplots(1,1)
    ax.set_aspect(aspect=1.)
    plt.tight_layout()
  
    # plot cluster space
    rgb_cols = np.reshape(data['f_rgb'][i,:,:,:]/(1.5*256.), [-1,data['f_rgb'].shape[-1]])
    alphas = np.expand_dims(np.sum(rgb_cols, axis=-1), axis=1)
    alphas = np.where(alphas < 0.95, np.zeros_like(alphas)+0.9,  np.zeros_like(alphas)+0.03)
    rgba_cols = np.concatenate([rgb_cols,alphas],axis=-1)
    
    betacols = np.reshape(data['p_beta'][i,:,:,0], [-1,1])
    betacols[betacols<0.01] = 0.01
    
    sorting = np.reshape(np.argsort(betacols, axis=0), [-1])
    
    
    #betacols -= np.min(betacols)
    #betacols /= np.max(betacols)
    print(np.max(betacols))
    rgbbeta_cols = np.concatenate([rgb_cols, betacols] ,axis=-1)
    
    ax.scatter(np.reshape(data['p_ccoords'][i,:,:,0], [-1])[sorting],
                  np.reshape(data['p_ccoords'][i,:,:,1], [-1])[sorting],
                  c=rgbbeta_cols[sorting])
    
    
    #ax[1][1].scatter(np.reshape(data['p_ccoords'][i][betaselection[i]][:,0], [-1]),
    #              np.reshape(data['p_ccoords'][i][betaselection[i]][:,1], [-1]),
    #              c=rgbbeta_cols)
    
    
    #for a in ax:
    #    a.set_aspect('equal')
    ax.set_aspect(1./ax.get_data_ratio())
    plt.tight_layout()
    fig.savefig("ccoords_"+str(i)+".pdf")
    plt.close()
    
    
    
    continue
    
    betapID = colourmask*data['p_ID']

    plt.imshow(betapID[i])
    fig.savefig("beta_pID"+str(i)+ ".png")
    
    
    
    
    
