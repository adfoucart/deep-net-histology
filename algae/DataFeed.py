import os
import numpy as np

class DataFeed:

    def __init__(self, dataset_path, patches_path, fuzzy_dataset=True, verbose=True):
        seed = 1

        self.v = verbose
        self.fuzzy_dataset = fuzzy_dataset
        if self.v: print("Loading dataset and patches")
        self.dataset = [os.path.join(dataset_path,f) for f in os.listdir(dataset_path) if f.find('.npy') > 0]
        self.patches = [os.path.join(patches_path,f) for f in os.listdir(patches_path) if f.find('.npy') > 0]
        if len(self.dataset) != len(self.patches):
            raise Exception("Dataset and patches should have the same number of files")

        # Get number of patches
        n_patches = []
        for f in self.dataset:
            a = np.load(f)
            n_patches += [a.shape[0]]
        n_patches = np.array(n_patches)
        if self.v: print("Total number of patches : %d"%n_patches.sum())

        file_ids = np.arange(len(self.dataset))

        if self.v: print("Using seed : %d"%seed)
        np.random.seed(seed) # Seed used to always use the same training set / test set for future reference
        np.random.shuffle(file_ids)

        # Find cutoff so that at most 80% of patches are in the training set
        cut_patches = int(n_patches.sum()*0.8)
        if self.v: print("Trying to get at most %d patches in training set"%cut_patches)
        t = 0
        self.training_set = []
        self.test_set = []
        for i in file_ids:
            t += n_patches[i]
            if t <= cut_patches:
                self.training_set += [i]
            else:
                self.test_set += [i]

        self.training_set = np.array(self.training_set)
        self.test_set = np.array(self.test_set)
        if self.v: print("Patches in training set : %d"%n_patches[self.training_set].sum())
        if self.v: print("Patches in test set : %d"%n_patches[self.test_set].sum())

        self.next_file = 0
        self.current_patch = None
        self.current_dataset = None
        self.pos_in_patch = 0

    def load_next_file(self):
        if self.v: print("Loading patches & dataset from next tile")
        i = self.training_set[self.next_file]
        try:
            self.current_patch = np.load(self.patches[i])
            self.current_dataset = np.load(self.dataset[i])
        except ValueError as e:
            print("i:",i)
            print("self.next_file:",self.next_file)
            print(self.patches[i])

        self.next_file = (self.next_file+1)%len(self.training_set)
        self.pos_in_patch = 0

    def fuzzy(self, Y):
        if self.fuzzy_dataset: return Y

        Y[Y==1.] = 0.7
        Y[Y==0.] = 0.1
        return Y

    def next_batch(self,batch_size):
        if self.current_patch is None:
            self.load_next_file()

        if( self.current_patch.shape[0]-batch_size-self.pos_in_patch < 0 ):
            if self.v: print("Between two files...")
            x0 = self.current_patch[self.pos_in_patch:]
            y0 = self.current_dataset[self.pos_in_patch:,4:]
            self.load_next_file()
            x1 = self.current_patch[:batch_size-x0.shape[0]]
            y1 = self.current_dataset[:batch_size-x0.shape[0],4:]
            self.pos_in_patch = batch_size-x0.shape[0]

            X = np.vstack([x0,x1])
            Y = np.vstack([y0,y1])
            if( X.shape[0] != batch_size ): return self.next_batch(batch_size)
            return X.astype('float32')-127,self.fuzzy(Y)

        X = self.current_patch[self.pos_in_patch:self.pos_in_patch+batch_size]
        Y = self.current_dataset[self.pos_in_patch:self.pos_in_patch+batch_size,4:]
        self.pos_in_patch += batch_size
        
        if( X.shape[0] != batch_size ): return self.next_batch(batch_size)
        return X.astype('float32')-127,self.fuzzy(Y)