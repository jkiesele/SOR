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
        for i in range(len(sorting[e])):
            use=True
            for s in selected:
                distance = math.sqrt( (s[0]-ccoords[e][i][0])**2 +  (s[1]-ccoords[e][i][1])**2 )
                if distance  < distance_threshold:
                    use=False
                    break
            if not use:
                betasel[e][i] = False
                continue
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


betaselection = collectoverthresholds(data, 0.1, 0.1)

print('betaselection',betaselection.shape)

def makeRectangle(size, pos):
    return patches.Rectangle([pos[1]-size[1]/2.,pos[0]-size[0]/2.],size[0],size[1],linewidth=1,edgecolor='y',facecolor='none')

selected_p_rectpos  = data['p_pos'][betaselection]
selected_p_rectpos = np.reshape(selected_p_rectpos, [selected_p_rectpos.shape[0], -1, selected_p_rectpos.shape[-1]])
selected_p_rectdims = data['p_dim'][betaselection]
selected_p_rectdims = np.reshape(selected_p_rectdims, [selected_p_rectdims.shape[0], -1, selected_p_rectdims.shape[-1]])


true_recspos = np.reshape(data['t_pos'], [data['t_pos'].shape[0],-1,data['t_pos'].shape[-1]]) 
true_recsdim = np.reshape(data['t_dim'], [data['t_dim'].shape[0],-1,data['t_dim'].shape[-1]])



    
for i in range(10):
    
    
    fig,ax = plt.subplots(2,2)
    
    
    
    ax[0][0].imshow(data['t_ID'][i])

    these_true_positions = []
    
    for j in range(len(true_recspos[i])):
        if (true_recspos[i][j][0],true_recspos[i][j][1]) in these_true_positions: continue
        these_true_positions.append((true_recspos[i][j][0],true_recspos[i][j][1]))
        
        rec = makeRectangle(true_recsdim[i][j], true_recspos[i][j] )
        ax[0][0].add_patch(rec)
    
    pIDimage = data['p_ID']
    pIDimage[np.invert(betaselection)] *=0
    ax[0][1].imshow(pIDimage[i])
    
    for j in range(len(selected_p_rectpos[i])):
        rec = makeRectangle(selected_p_rectdims[i][j], selected_p_rectpos[i][j] )
        ax[0][1].add_patch(rec)
    
  
    # plot cluster space
    
    
    ax[1][0].scatter(np.reshape(data['p_ccoords'][i,:,:,0], [-1]),
                  np.reshape(data['p_ccoords'][i,:,:,1], [-1]),
                  c=np.reshape(data['t_ID'][i,:,:,1], [-1]), alpha=0.5)
    
    
    ax[1][1].scatter(np.reshape(data['p_ccoords'][i][betaselection[i]][:,0], [-1]),
                  np.reshape(data['p_ccoords'][i][betaselection[i]][:,1], [-1]),
                  c=np.reshape(data['t_ID'][i][betaselection[i]][:,1], [-1]), alpha=0.5)
    
    
    #for a in ax:
    #    a.set_aspect('equal')
    
    fig.savefig("sel_ccoords"+str(i)+ ".png")
    
    
    
    continue
    
    betapID = colourmask*data['p_ID']

    plt.imshow(betapID[i])
    fig.savefig("beta_pID"+str(i)+ ".png")
    
    
    
    
    