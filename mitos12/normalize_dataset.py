import numpy as np
import os

dspath = 'e:/data/MITOS12/train/batches'
ds = [os.path.join(dspath, f) for f in os.listdir(dspath) if f.find('batch') >= 0]

for f in ds:
	batch = np.load(f)

	batch[:33,:,:] *= 255
	a = batch.mean(axis=1).mean(axis=1).mean(axis=1)

	for b in range(200):
	    batch[b] = batch[b] - a[b]

	np.save(f, batch)
	print(f)