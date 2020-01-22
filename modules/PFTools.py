

#
#
# 2 inputs: one calo, one tracker
# calo:    16x16
# tracker: 64x64
#
# use trainData as input. feat[0] and feat[1]
#
# inputs are: energy, x, y
#
#

import numpy as np
from numba import jit

import math


E_seed = 0.230
E_cell = 0.080
sigma = 15 #(everything mm)

class calo_cluster(object):
    def __init__(self,energy,pos,true_energy=-1,true_pos=[-1000,-1000]):
        #self.seed_idx=[]
        self.energy=float(energy)
        self.position=pos
        self.true_energy=float(true_energy)
        self.true_position=true_pos
        
    def __str__(self):
        outstr="CaloCluster: energy "+str(self.energy)+"+-"+str(self.rel_resolution()*self.energy)+" GeV, pos "+str(self.position)
        outstr+=" truth energy " +str(self.true_energy)+" truth pos "+str(self.true_position)
        return outstr
    
    def rel_resolution(self):
        a = 0.028/math.sqrt(self.energy)
        b = 0.12/self.energy
        c = 0.003
        return math.sqrt(a**2 +b**2 + c**2)
    
    def corrected_energy(self):
        one_over_factor=0.847919 + 0.00639966*self.energy + -0.000404986*self.energy**2 \
        + 1.10737e-05*self.energy**3 + -1.57167e-07*self.energy**4 + 1.20291e-09*self.energy**5\
        + -4.71213e-12*self.energy**6 + 7.41855e-15*self.energy**7
        return  self.energy/one_over_factor #+ 7.2063e-06*self.energy**3



@jit(nopython=True)     
def gen_get_truth_particles(eventtruth):#truth input: V x F
    truthpos    =eventtruth[:,2:4]
    truthenergy = eventtruth[:,1:2]
    tpos=[]
    ten =[]
    for i in range(truthpos.shape[0]):
        en = truthenergy[i][0]
        if en==0 or en in ten: continue
        pos = [truthpos[i][0],truthpos[i][1]]
        tpos.append(pos)
        ten.append(en)
    return np.array(tpos),np.array(ten)
     

#
# Add the tracks here in the sense that:
# If not seed, but track in front (within 2.2/2 cm) -> make seed
#
@jit(nopython=True)     
def c_calo_getSeeds(caloinput,seed_idxs,trackerinput,map):
    #get tracker hits first
    # and mark as seeds whatever calo is behind
    #make the mask:
    trackgrid_position = trackerinput[0,:,:,1:3]#positions
    calogrid_position = caloinput[0,:,:,1:3]#positions
    #make a map between tracker and calo
    for xc in range(calogrid_position.shape[0]):
        for yc in range(calogrid_position.shape[1]):
            for xt in range(trackgrid_position.shape[0]):
                for yt in range(trackgrid_position.shape[1]):
                    diffx = abs(trackgrid_position[xt,yt,0]-calogrid_position[xc,yc,0])
                    diffy = abs(trackgrid_position[xt,yt,1]-calogrid_position[xc,yc,1])
                    if diffx < 22/2. and diffy < 22/2.:
                        map[xt,yt,0]=xc
                        map[xt,yt,1]=yc
                    
            
    
    
    for ev in range(trackerinput.shape[0]):
        for xt in range(trackerinput.shape[1]):
            for yt in range(trackerinput.shape[2]):
                if trackerinput[ev,xt,yt,0] > 0.1:
                    seed_idxs[ev,map[xt,yt,0],map[xt,yt,1]]=1
        
        
    
    
    
    xmax=caloinput.shape[1]
    ymax=xmax #symmetric
    for ev in range(len(caloinput)):
        for x in range(len(caloinput[ev])):
            for y in range(len(caloinput[ev][x])):
                seed_e = caloinput[ev,x,y,0]
                if seed_e < E_seed:
                    continue
                #check surrounding cells
                is_seed=True
                for xs in range(x-1,x+2):
                    if xs<0 or xs>=xmax: continue
                    for ys in range(y-1,y+2):
                        if ys<0 or ys>=ymax: continue
                        if xs == x and ys ==y: continue
                        if caloinput[ev,xs,ys,0]>=seed_e:
                            is_seed = False
                        if not is_seed: break
                    if not is_seed: break
                if is_seed:
                    seed_idxs[ev,x,y]=1
    outseedidxs=[]
    for ev in range(len(caloinput)):
        outseedidxs.append(seed_idxs[ev]>0)
    return outseedidxs, seed_idxs

def calo_getSeeds(caloinput,trackerinput):
    seed_idxs=np.zeros( [caloinput.shape[0],caloinput.shape[1],caloinput.shape[2]] )
    map = np.zeros((trackerinput.shape[1],trackerinput.shape[2],2), dtype='int64')-1
    sidxl, seedids = c_calo_getSeeds(caloinput,seed_idxs,trackerinput,map)
    return sidxl, seedids


def calo_calc_f(A,c,mu):
    #expand i: aka the seed dim to axis0
    A_exp  = np.expand_dims(A, axis=0)  # 1 x N 
    mu_exp = np.expand_dims(mu, axis=0) # 1 x N x 2
    c_exp  = np.expand_dims(c, axis=1)  # M x 1 x 2
    
    posdelta = np.sum((c_exp-mu_exp)**2, axis=-1) # M x N
    
    upper = A_exp * np.exp(- posdelta/(2*sigma**2)) # M x N
    lower = np.sum( upper, axis=1, keepdims=True )
    
    return upper/(lower + 1e-7)
    
def calo_calc_A(f,E):# E: M , f: M x N
    E_exp = np.expand_dims(E, axis=1) # M x 1
    A = np.sum( f*E_exp , axis=0) # N
    return A
    
def calo_calc_mu(f,c,E): #c: M x 2, f: M x N, E: M 
    c_exp = np.expand_dims(c, axis=1) # M x 1 x 2
    f_exp = np.expand_dims(f, axis=2) # M x N x 1
    E_exp = np.expand_dims(np.expand_dims(E, axis=1), axis=1) # M x 1 x 1
    
    den = np.sum(f_exp*E_exp,axis=0)
    mu = np.sum(f_exp*E_exp*c_exp, axis=0) / den # N x 2
    return mu
    
    
def calo_determineClusters(caloinput, seedidxs):#calo input per event: x X y X F  /// seedidxs: also per event
    
    energies = np.reshape(caloinput[:,:,0:1], [-1,1]) #keep dim
    energies[energies<E_cell] = 0
    allhits = np.concatenate([energies, np.reshape(caloinput[:,:,1:], [caloinput.shape[0]**2, -1])],axis=-1)
    
    seed_properties = caloinput[seedidxs] #now 
    if len(seed_properties)<1:
        return []
    seed_energies = seed_properties[:,0]
    seed_pos = seed_properties[:,1:3]
    
    #make the matrices
    E = allhits[:,0]
    c = np.array(allhits[:,1:3]) 
    
    A = np.array(seed_energies) #initial
    mu = np.array(seed_pos) #initial
    not_converged=True
    f = calo_calc_f(A, c, mu)
    
    counter=0
    while(not_converged and counter<100):
        #use initial A for now
        new_mu = calo_calc_mu(f,c,E)
        mmudiff = np.max( np.sum((new_mu-mu)**2,axis=-1) )
        #print(mmudiff)
        if mmudiff < .2**2:
            not_converged = False
        mu = new_mu
        f = calo_calc_f(A, c, mu)
        counter+=1
    print(counter)
    A = calo_calc_A(f,E)
    
    out=[]
    for i in range(len(A)):
        if not np.isnan(A[i]): #fit failed
            out.append(calo_cluster(A[i],mu[i]))
    
    return out
    

#@jit(nopython=True)     
def c_match_cluster_to_truth(clusters, true_pos, true_en, truth_used):# per event, true_pos: sequence. find truth for reco
    for cl in clusters:
        L1_dist=10000
        best_it=-1
        for i_t in range(len(true_pos)):
            if truth_used[i_t]: continue
            this_L1_dist = abs(cl.position[0]-true_pos[i_t][0])+abs(cl.position[0]-true_pos[i_t][0])
            if this_L1_dist < 22. and this_L1_dist < L1_dist:
                 best_it=i_t
                 L1_dist=this_L1_dist
        if best_it>-1:
            truth_used[best_it]=1
            cl.true_energy = true_en[best_it]
            cl.true_position[0] = true_pos[best_it][0]
            cl.true_position[1] = true_pos[best_it][1]
    
    
    return clusters
    
def match_cluster_to_truth(clusters, true_pos, true_en):
    truth_used = np.zeros_like(true_en, dtype='int64')
    return c_match_cluster_to_truth(clusters, true_pos, true_en, truth_used)
    


def perform_linking(clusters, tracks):
    valid_tracks = tracks[tracks[:,:,0]>0.1]
    









    