import os
import numpy as np

class EqualizedDataFeed:

    def __init__(self, dataset_path, verbose=True):
        seed = 1

        self.v = verbose
        if self.v: print("Loading targets and patches")
        self.batch_targets = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path) if f.find('_target.npy') > 0]
        self.batch_patches = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path) if f.find('_patch.npy') > 0]
        if len(self.batch_targets) != len(self.batch_patches):
            raise Exception("Target and patches should have the same number of files")

        n_batches = len(self.batch_targets)
        batch_ids = np.arange(n_batches)

        if self.v: print("Using seed : %d"%seed)
        np.random.seed(seed) # Seed used to always use the same training set / test set for future reference
        np.random.shuffle(batch_ids)

        # Find cutoff so that at most 80% of patches are in the training set
        cut_patches = int(n_batches*0.8)
        if self.v: print("Trying to get at most %d patches in training set"%cut_patches)
        t = 0

        self.training_set = batch_ids[:cut_patches]
        self.test_set = batch_ids[cut_patches:]

        if self.v: print("Patches in training set : %d"%len(self.training_set))
        if self.v: print("Patches in test set : %d"%len(self.test_set))

        self.cur_epoch = 0
        self.cur_batch = 0
    
    def random_symmetry(self, X):
        do_v_sym_on = np.random.random((X.shape[0],))>0.5
        do_h_sym_on = np.random.random((X.shape[0],))>0.5
        X[do_v_sym_on] = X[do_v_sym_on,::-1,:,:]
        X[do_h_sym_on] = X[do_h_sym_on,:,::-1,:]
        return X

    def fuzzy(self, Y):
        Y[Y==1.] = 0.7
        Y[Y==0.] = 0.1
        return Y

    def next_batch(self,batch_size):
        if batch_size != 200:
            raise Exception("EqualizedDataFeed requires a batch size of 200")

        if self.cur_batch >= len(self.training_set):
            self.cur_batch = 0
            self.cur_epoch += 1
            if self.v: print("Starting epoch %d"%self.cur_epoch)

        fid = self.training_set[self.cur_batch]
        X = self.random_symmetry(np.load(self.batch_patches[fid]).astype('float32')-127)
        Y = self.fuzzy(np.load(self.batch_targets[fid]))

        return X,Y