


import numpy as np
from numba import jit

import math

from inference import make_particle_inference_dict

'''
outdict['t_mask'] =  truth[:,:,0:1]
        outdict['t_E']    =  truth[:,:,1:2]
        outdict['t_pos']  =  truth[:,:,2:4]
        outdict['t_ID']   =  truth[:,:,4:6]
        outdict['t_objidx']= truth[:,:,6:7]
        
        outdict['t_rhpos']= truth[:,:,7:10]
        outdict['t_rhid']= truth[:,:,10:11]
'''

tracker_pos_res = 22./4.

#matches reco to truth, per event
@jit(nopython=True)     
def c_find_best_matching_truth(pf_pos, pf_energy, t_Mask, t_pos, t_E, t_objidx,t_ID, rest_t_objidx, pos_tresh):
    # flags any used truth index with -1
    # returns per pf matched truth info. for non matched, returns -1
    
    matched_id = []
    matched_e = []
    matched_pos = []
    
    
    for i_pf in range(len(pf_pos)):
        pf_x = pf_pos[i_pf][0]
        pf_y = pf_pos[i_pf][1]
        pf_e = pf_energy[i_pf]
        
        
        bestmatch_idx=-1
        bestmatch_distance_sq = 2*pos_tresh**2+1e6
        bestmatch_energy_diff=1000
        bestmatch_energy = -1.
        bestmatch_posx = -999.
        bestmatch_posy = -999.
        bestmatch_id = -1
        
        for i_t in range(len(t_pos)):
            t_x = t_pos[i_t][0]
            t_y = t_pos[i_t][1]
            t_e = t_E[i_t][0]
            t_mask = t_Mask[i_t][0]
            if t_mask < 1:
                continue #noise
            t_idx = t_objidx[i_t][0]
            t_id = t_ID[i_t][0]
            
            
            dist_sq = (pf_x-t_x)**2 + (pf_y-t_y)**2
            abs_energy_diff = abs(t_e - pf_e)
            
            if dist_sq > pos_tresh**2 : continue
            
            #get to about 22^2 contribution for 5 % relative momentum difference (3 sigma-ish)
            dist_sq += (22./0.1)**2 * (pf_e/t_e -1)**2
            
            #if abs(dist_sq - bestmatch_distance_sq) < 0.001**2: #same position check for energy
            #    if bestmatch_energy_diff < abs_energy_diff: continue
            if bestmatch_distance_sq < dist_sq : continue
                
            bestmatch_distance_sq = dist_sq
            bestmatch_energy_diff = abs_energy_diff
            bestmatch_idx = t_idx
            bestmatch_energy = t_e
            bestmatch_posx = t_x
            bestmatch_posy = t_y
            bestmatch_id = t_id
         
        matched_id.append(bestmatch_id)    
        matched_e.append(bestmatch_energy)
        matched_pos.append([bestmatch_posx,bestmatch_posy])
        if bestmatch_idx>=0: #truth match
            for i_iidx in range(len(rest_t_objidx)):
                if bestmatch_idx == rest_t_objidx[i_iidx]:
                    rest_t_objidx[i_iidx] = -1
            
        
            
    not_recoed_pos=[]
    not_recoed_e=[] 
    not_recoed_id=[]
    #get truth for non matched
    for t_i in rest_t_objidx:
        if t_i < 0: continue
        for i_t in range(len(t_pos)):
            t_idx = t_objidx[i_t][0]
            t_mask = t_Mask[i_t][0]
            if t_mask < 1:
                continue #noise
            if not abs(t_idx - t_i)<0.1: continue
            t_x = t_pos[i_t][0]
            t_y = t_pos[i_t][1]
            t_e = t_E[i_t][0]
            t_id = t_ID[i_t][0]
            not_recoed_pos.append([t_x,t_y])
            not_recoed_e.append(t_e)
            not_recoed_id.append(t_id)
            break
            
            
        
            
    return matched_pos, matched_e, matched_id, not_recoed_pos, not_recoed_e, not_recoed_id
    
    
def multi_expand_dims(alist, axis):
    out=[]
    for a in alist:
        out.append(np.expand_dims(a,axis=axis))
    return out  


def make_evaluation_dict(array):
    #is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id
    out={}
    out['is_reco'] = array[:,0]
    out['reco_posx'] = array[:,1]
    out['reco_posy'] = array[:,2]
    out['reco_e'] = array[:,3]
    out['is_true'] = array[:,4]
    out['true_posx'] = array[:,5]
    out['true_posy'] = array[:,6]
    out['true_e'] = array[:,7]
    out['true_id'] = array[:,8]
    
    return out
    
    #is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id


#takes one event
def find_best_matching_truth_and_format(pf_pos, pf_energy, truth, pos_tresh=22.): #two ecal cells
    
    '''
    returns p particle: is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id
    '''
    
    d = make_particle_inference_dict(None, None, np.expand_dims(truth,axis=0))
    
    
    rest_t_objidx = np.unique(d['t_objidx'][0])
    n_true = float(len(rest_t_objidx)-1.)
    matched_pos, matched_e, matched_id, not_recoed_pos, not_recoed_e, not_recoed_id = c_find_best_matching_truth(pf_pos, 
                                                                     pf_energy, 
                                                                     d['t_mask'][0],d['t_pos'][0], d['t_E'][0], d['t_objidx'][0], d['t_ID'][0],
                                                                     rest_t_objidx, pos_tresh)
    
    
    
    matched_posx,matched_posy    = np.array(matched_pos)[:,0], np.array(matched_pos)[:,1]
    matched_e      = np.array(matched_e)
    matched_id     = np.array(matched_id)
    
    is_reco = np.zeros_like(matched_e)+1
    is_true = np.where(matched_e>=0, np.zeros_like(matched_e)+1., np.zeros_like(matched_e))
    
    n_true_arr = np.tile(n_true,[matched_e.shape[0]])
    #print('is_reco',is_reco.shape)
    #print('pf_pos',pf_pos.shape)
    #print('pf_energy',pf_energy.shape)
    #print('is_true',is_true.shape)
    #print('matched_posx',matched_posx.shape)
    #print('matched_posy',matched_posy.shape)
    #print('matched_e',matched_e.shape)
    #print('matched_id',matched_id.shape)
    
    all_recoed = multi_expand_dims([is_reco, pf_pos[:,0],pf_pos[:,1], pf_energy, 
                                 is_true, matched_posx,matched_posy, matched_e, matched_id, n_true_arr],axis=1)
    #concat the whole thing
    all_recoed = np.concatenate(all_recoed,axis=-1)
    
    
    if len(not_recoed_e):
        not_recoed_posx, not_recoed_posy = np.array(not_recoed_pos)[:,0], np.array(not_recoed_pos)[:,1]
        not_recoed_e   = np.array(not_recoed_e)
        not_recoed_id  = np.array(not_recoed_id)
        n_true_arr = np.tile(n_true,[not_recoed_e.shape[0]])
         
        all_not_recoed = multi_expand_dims([np.zeros_like(not_recoed_e)+1., not_recoed_posx, not_recoed_posy, not_recoed_e, not_recoed_id,n_true_arr], axis=1)
        #is_true, true_posx, true_posy, true_e, true_id
        all_not_recoed = np.concatenate(all_not_recoed,axis=-1)
        all_not_recoed = np.pad(all_not_recoed, [(0,0),(4,0)], mode='constant', constant_values=0)
        all_recoed = np.concatenate([all_recoed,all_not_recoed],axis=0)
    
    # particle: is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id
    return all_recoed
   
    
    
def write_output_tree(allparticles, outputFile):
    from root_numpy import array2root
    out = np.core.records.fromarrays(allparticles.transpose() ,names="is_reco, reco_posx, reco_posy, reco_e, is_true, true_posx, true_posy, true_e, true_id, n_true")
    array2root(out, outputFile+".root", 'tree')    
    
    
    
    