#!/usr/bin/env python


from __future__ import print_function
import numpy as np
from argparse import ArgumentParser

from evaluation_tools import find_best_matching_truth_and_format, write_output_tree
from PFTools import calo_getSeeds, calo_determineClusters, perform_linking, create_pf_tracks


parser = ArgumentParser('Performs standard PF reco and puts output into root file')
parser.add_argument('inputFile')
parser.add_argument('outputFile')

args=parser.parse_args()

from DeepJetCore.TrainData import TrainData


allparticles=[]

with open(args.inputFile) as file:
    for inputfile in file:
        inputfile = inputfile.replace('\n', '')
        if len(inputfile)<1: continue
        
        td = TrainData()
        td.readFromFile(inputfile)
        feat = td.transferFeatureListToNumpy()
        truth = td.transferTruthListToNumpy()[0]
        del td
        calo = feat[0]
        tracks = feat[1]
        
        
        print('making seeds, also using tracks as seeds')
        seed_idxs, seedmap = calo_getSeeds(calo,tracks)
        
        for event in range(len(calo)):
            
            #make calo clusters
            print('event',event)
            
            e_seeds = seed_idxs[event]
            e_calo = calo[event,:,:,:]
            e_tracks = tracks[event,:,:,:]
            
            PF_calo_clusters = calo_determineClusters(e_calo, e_seeds)
            PF_tracks = create_pf_tracks(e_tracks)
            
            pfcands = perform_linking(PF_calo_clusters,PF_tracks)
        
            ev_reco_E    = np.array([p.energy for p in pfcands])
            ev_reco_pos  = np.array([p.position for p in pfcands])
            ev_truth = truth[event]
            
            
            eventparticles = find_best_matching_truth_and_format(ev_reco_pos, ev_reco_E, ev_truth, pos_tresh=2*22.)
            allparticles.append(eventparticles)
    
allparticles = np.concatenate(allparticles,axis=0)

print('efficiency: ', float(np.count_nonzero( allparticles[:,0] *  allparticles[:,4]))/float( np.count_nonzero(allparticles[:,4] ) ))
print('fake: ', float(np.count_nonzero( allparticles[:,0] *  (1.-allparticles[:,4])))/float( np.count_nonzero(allparticles[:,0] ) ))

np.save(args.outputFile+".npy", allparticles)
write_output_tree(allparticles, args.outputFile+".root")


