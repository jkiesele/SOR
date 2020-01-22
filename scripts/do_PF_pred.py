#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from DeepJetCore.TrainData import TrainData
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PFTools import calo_getSeeds, calo_determineClusters, gen_get_truth_particles, match_cluster_to_truth


parser = ArgumentParser('perform PF reco')
parser.add_argument('inputFile')

args=parser.parse_args()

td = TrainData()
td.readFromFile(args.inputFile)

feat = td.transferFeatureListToNumpy()
truth = td.transferTruthListToNumpy()[0]

calo = feat[0]
tracker = feat[1]

print(calo.shape)
print(tracker.shape)

debug = False

print('making seeds')
seed_idxs, seedmap = calo_getSeeds(calo,tracker)


allclusters=[]
# save the output in same format as the DNN prediction
print('creating clusters by event')
for event in range(len(calo)):
    #print calo for event 0
    e_seeds = seed_idxs[event]
    #print('e_seeds',e_seeds)
    thisevent = calo[event,:,:,:]
    #print(thisevent)
    if debug:
        
        fig,ax = plt.subplots(1,1)
        ax.scatter(x= np.reshape(thisevent[:,:,1],-1), y=np.reshape(thisevent[:,:,2],-1),
                   c = np.reshape(thisevent[:,:,0],-1))
        
        fig.savefig("pftest"+str(event)+".pdf")
        
        ax.imshow(thisevent[:,:,0]/np.max(thisevent[:,:,0]))
        
        fig.savefig("pftest2_"+str(event)+".pdf")
        
        ax.imshow(seedmap[event])
        fig.savefig("seedmap"+str(event)+".pdf")
        
    
    seeds = thisevent[e_seeds]
    
    seedpos= thisevent[e_seeds][:,1:3]
    
    tpos, ten = gen_get_truth_particles(truth[event])
    
    
    #print(seedpos)
    
    
    PF_calo_clusters = calo_determineClusters(thisevent, e_seeds)
    
    PF_calo_clusters = match_cluster_to_truth(PF_calo_clusters,tpos,ten)
    if len(PF_calo_clusters):
        allclusters+=PF_calo_clusters
    
    
    
    print('PF_calo_clusters', [str(cl) for cl in PF_calo_clusters])
    
    
clust_energies =np.array([cl.energy for cl in allclusters])
corr_clust_energies=np.array([cl.corrected_energy() for cl in allclusters])
true_energies  =np.array([cl.true_energy for cl in allclusters])

from root_numpy import array2root
out = np.core.records.fromarrays([clust_energies,corr_clust_energies, true_energies],names='clust_energies,corr_clust_energies, true_energies')
array2root(out, "pf_cluster_calibration.root", 'tree')

    
    
    
    