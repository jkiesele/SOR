
from __future__ import print_function
import numpy as np
from DeepJetCore.TrainData import TrainData
from argparse import ArgumentParser

from PFTools import calo_getSeeds, calo_determineClusters


parser = ArgumentParser('Performs standard PF reco and puts output into root file')
parser.add_argument('inputFile')
parser.add_argument('outputFile')


args=parser.parse_args()

td = TrainData()
td.readFromFile(args.inputFile)
feat = td.transferFeatureListToNumpy()
truth = td.transferTruthListToNumpy()[0]
del td
calo = feat[0]
tracker = feat[1]


print('making seeds')
seed_idxs, seedmap = calo_getSeeds(calo,tracker)

for event in range(len(calo)):
    
    #make calo clusters
    
    e_seeds = seed_idxs[event]
    thisevent = calo[event,:,:,:]
    
    PF_calo_clusters = calo_determineClusters(thisevent, e_seeds)
    
    # perform track linking
    
    
    #build pf candidates
    
    
    # format to prediction, mask, feat, truth
    # 