

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
cellsize = 22 #(everything mm)

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


class pf_track(object):
    def __init__(self,energy,pos):
        self.energy=energy
        self.position=pos
        
    def rel_resolution(self):
        pt=self.energy
        return (pt/100.)*(pt/100.)*0.04 +0.01;


class pfCandidate(calo_cluster):
    def __init__(self,energy=-1,pos=[0.,0.]):
        calo_cluster.__init__(self,energy,pos)
        
    
    def create_from_link(self, calocluster, pftrack=None):
        if pftrack is None:
            self.energy=calocluster.energy
            self.position=calocluster.position
            return None
        else: #use a weighted mean
            
            self.position = pftrack.position
            
            t = pftrack.energy
            dt = pftrack.rel_resolution()
            c = calocluster.energy
            dc = calocluster.rel_resolution()
            
            if abs(t-c) > math.sqrt((c*dc)**2+(t*dt)**2): #incompatible, create second pf candidate
                self.energy = t #use track momentum
                neutral_cand = pfCandidate(c-t, calocluster.position)
                if c-t > 0.5:
                    return neutral_cand
                
            self.energy = 1/((t*dt)**(-2) + (c*dc)**(-2)) * (c/(c*dc)**2 +t/(t*dt)**2)
            
            return None
        


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
    #print(counter)
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
            this_L1_dist = abs(cl.position[0]-true_pos[i_t][0])+abs(cl.position[1]-true_pos[i_t][1])
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
    
    
def create_pf_tracks(tracks):
    #input tracks[event,:,:,:] 0: energy, 1,2 pos
    out_tracks = []
    intracks = np.reshape(tracks, [-1,tracks.shape[-1]])
    for i_t in range(len(intracks)):
        if intracks[i_t][0] < 1: continue
        out_tracks.append(pf_track(intracks[i_t][0],intracks[i_t,1:3]))
    return out_tracks

@jit(nopython=True)  
def c_perform_linking(cluster_positions, track_positions):
    
    matching=[] #i_cluster, i_track
    for i_c in range(len(cluster_positions)):
        best_distancesq = 1e3
        best_track = -1
        for i_t in range(len(track_positions)):
            distsq = (cluster_positions[i_c][0]-track_positions[i_t][0])**2 + (cluster_positions[i_c][1]-track_positions[i_t][1])**2
            if distsq > cellsize: continue
            if best_distancesq < distsq: continue
            best_track = i_t
            best_distancesq = distsq
        matching.append([i_c, best_track])
        
    return matching
    
#works on an event by event basis
#clusters are calo_cluster objects, tracks are pf_track objects
#returns list of candidates directly
def perform_linking(clusters, tracks):
    
    cluster_positions = np.array([c.position for c in clusters])
    track_positions = np.array([t.position for t in tracks])
    matching = c_perform_linking(cluster_positions, track_positions)
    
    particles = []
    for m in matching:
        pfc = pfCandidate()
        cand2=None
        if m[1]>=0:
            cand2 = pfc.create_from_link(clusters[m[0]], tracks[m[1]])
        else:
            pfc.create_from_link(clusters[m[0]])
            
        particles.append(pfc)
        if cand2 is not None:
            particles.append(cand2)
            
    return particles
    
    
    
    
    

    





    