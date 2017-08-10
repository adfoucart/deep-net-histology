import numpy as np
import os

basepath = "e:/data/MITOS12/train/batches/"
files = os.listdir(basepath)
files_patch = [f for f in files if f.find('patches') >= 0]
files_target = [f for f in files if f.find('target') >= 0]

n_with_stuff = 0
t = np.zeros((20,127,127))
for i,f in enumerate(files_target):
    t[:,:,:] = np.load(os.path.join(basepath, f))
    stuff_in = t.max(axis=1).max(axis=1) > 0
    
    n_with_stuff += stuff_in.sum()
    if i%100 == 0 :
        print(i)