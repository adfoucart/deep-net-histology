import numpy as np
import os

class MITOS12Feed:

	def __init__(self, **kwargs):
		self.training_dir = kwargs['training_dir'] if 'training_dir' in kwargs else 'e:/data/MITOS12/train/batches'

		self.batch_files = [os.path.join(self.training_dir, f) for f in os.listdir(self.training_dir) if f.find('batch') >= 0]
		self.target_files = [os.path.join(self.training_dir, f) for f in os.listdir(self.training_dir) if f.find('target') >= 0]

		self.n_files = len(self.batch_files)
		if(self.n_files != len(self.target_files)):
			print("Error : %d batch files and %d target files in training dir"%(self.n_files, len(self.target_files)))

		self.idx = np.arange(self.n_files)
		np.random.seed(0)
		np.random.shuffle(self.idx)

		self.current = 0

	def next_batch(self):
		X = np.load(self.batch_files[self.idx[self.current]])
		Y = np.load(self.target_files[self.idx[self.current]])
		self.current += 1
		if self.current >= self.n_files : self.current = 0

		return X,Y