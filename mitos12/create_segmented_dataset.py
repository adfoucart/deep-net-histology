''' We now want to keep the location of the target, not just the probability. If we make the input a w x h RGB tile, the target will be a w x h boolean tile. '''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches as patches
import csv
import os

def get_bounding_boxes(super_file):
	with open(super_file) as f:
		reader = csv.reader(f, delimiter=',')
		mitosis_raw = [np.array(row).astype('int') for row in reader]
		mitosis_coord = [np.vstack([m[::2], m[1::2]]) for m in mitosis_raw]
		
		mitosis_bbox = np.vstack([np.hstack([m.min(axis=1), m.max(axis=1)]) for m in mitosis_coord])
		return mitosis_bbox
	return None

def get_mask(super_file):
	mask = np.zeros((2084,2084))
	with open(super_file) as f:
		reader = csv.reader(f, delimiter=',')
		mitosis_raw = [np.array(row).astype('int') for row in reader]
		mitosis_coord = [np.vstack([m[::2], m[1::2]]) for m in mitosis_raw]
		for c in mitosis_coord:
			mask[c[1],c[0]] = c.shape[1]
		return mask
	return None

def generate_patches(fname, supervision_fname, basedir, tilesize=127, stride=10, withTargets=True):
	im = plt.imread(os.path.join(basedir, fname))
	mask = get_mask(os.path.join(basedir, supervision_fname))

	print("Generating patches for %s..."%fname)
	rangex = np.arange(0,im.shape[0]-tilesize,stride)
	rangey = np.arange(0,im.shape[1]-tilesize,stride)
	ts = [(t//len(rangey), t%len(rangey)) for t in range(len(rangex)*len(rangey))]
	chunks = [[tx*stride+tilesize//2,ty*stride+tilesize//2] for tx,ty in ts] # Centers
	n_per_batch = int(np.sqrt(len(chunks)))

	for b in range(n_per_batch):
		# patches = np.zeros((n_per_batch,tilesize,tilesize,3))
		targets = np.zeros((n_per_batch,tilesize,tilesize))
		for i,c in enumerate(chunks[b*n_per_batch:(b+1)*n_per_batch]):
			# patches[i] = im[c[0]-tilesize//2:c[0]+tilesize//2+1,c[1]-tilesize//2:c[1]+tilesize//2+1]
			targets[i] = mask[c[0]-tilesize//2:c[0]+tilesize//2+1,c[1]-tilesize//2:c[1]+tilesize//2+1]
		# np.save(os.path.join(basedir, 'patches/%s.%04d.npy'%(fname,b)), patches)
		if withTargets: np.save(os.path.join(basedir, 'targets/%s.%04d.npy'%(fname,b)), targets)

def transform_batch(X, Y, params):
	X2 = np.zeros((len(X),127,127,3))
	Y2 = np.zeros((len(Y),127,127))
	# Noise, illumination & translation
	for i in range(len(X)):
		X2[i,:,:,:] = X[i,int(params[i,0]):int(params[i,0])+127,int(params[i,1]):int(params[i,1])+127,:]+(np.random.random(X2[i].shape)*params[i,3]-params[i,3]/2)+params[i,4]
		Y2[i,:,:] = Y[i,int(params[i,0]):int(params[i,0])+127,int(params[i,1]):int(params[i,1])+127]

	# Orientation
	do_vswap = (params[:,2]==1)+(params[:,2]==3)
	do_hswap = (params[:,2]==2)+(params[:,2]==3)
	X2[do_vswap] = X2[do_vswap,::-1,:,:]
	X2[do_hswap] = X2[do_hswap,:,::-1,:]
	Y2[do_vswap] = Y2[do_vswap,::-1,:]
	Y2[do_hswap] = Y2[do_hswap,:,::-1]

	return X2,Y2

def get_random_transformation(n):
	params = np.zeros((n,5))
	params[:,0] = (np.random.random((n,))*40).astype('int')
	params[:,1] = (np.random.random((n,))*40).astype('int')
	params[:,2] = np.floor(np.random.random((n,))*4).astype('int')
	params[:,3] = np.random.random((n,))*0.2
	params[:,4] = np.random.random((n,))*0.2 - 0.1
	return params

# Get all files
basedir = "e:/data/MITOS12/train/"
targetsdir = os.path.join(basedir,'targets')
patchesdir = os.path.join(basedir,'patches')
#batchesdir = os.path.join(basedir,'batches')
image_files = [f for f in os.listdir(basedir) if f.find('.bmp') > 0]
super_files = [f for f in os.listdir(basedir) if f.find('.csv') > 0]
target_files = [os.path.join(targetsdir,f) for f in os.listdir(targetsdir)]
patches_files = [os.path.join(patchesdir,f) for f in os.listdir(patchesdir)]

#for i in range(len(image_files)):
# 	generate_patches(image_files[i], super_files[i], basedir, 167, 30, True)	# 167- tile so that we can do some random translation

# Transform & generate batches
Tidx = np.load('e:/data/MITOS12/train/train_mitosis_indexes.npy')
idref = np.arange(len(Tidx))
mitosis_idx = idref[Tidx==True]
nothing_idx = idref[Tidx==False]

idx2fl = lambda idx : (idx//64, idx%64)

n_batches = 50000

patches = np.zeros((20,127,127,3))
targets = np.zeros((20,127,127))
	
for i in range(n_batches):
	X = np.zeros((20,167,167,3))
	Y = np.zeros((20,167,167))
	np.random.shuffle(mitosis_idx)
	np.random.shuffle(nothing_idx)
	idxs = np.hstack([mitosis_idx[:10],nothing_idx[:10]])
	fl = [idx2fl(idx) for idx in idxs]

	for j,(f,l) in enumerate(fl):
		X[j] = np.load(patches_files[f])[l]/255.
		Y[j] = np.load(target_files[f])[l]
	
	params = get_random_transformation(20)
	patches,targets = transform_batch(X,Y,params)
	np.save('e:/data/MITOS12/train/batches/patches.%05d.npy'%i,patches)
	np.save('e:/data/MITOS12/train/batches/targets.%05d.npy'%i,targets)

	print("%.2f %%"%(i*100/n_batches),end='\r')

'''# stats_on_training_set(basedir)
with_mitosis,no_mitosis = get_sets_with_mitosis(targetsdir)
# create_transformation_matrix(os.path.join(basedir,'patches'))
transformation_matrix = np.load(os.path.join(basedir, 'mitos12_full_transformations_matrix.npy'))

n_somemitosis = 0
id_to_patch = {}
id_ref = np.arange(196)
for wm in with_mitosis:	# for each file with at least some mitosis
	for i in range(len(with_mitosis[wm][with_mitosis[wm]])):	# loop through all patches with some mitosis
		id_to_patch[n_somemitosis+i] = {'file':wm, 'id': id_ref[with_mitosis[wm]][i]}	# with_mitosis[wm] -> 0 where there is no mitosis, 1 elsewhere -> id_ref[with_mitosis[wm]] = ids of all patches with some mitosis
	n_somemitosis += len(with_mitosis[wm][with_mitosis[wm]])

# p = load_patch(patchesdir, id_to_patch[4])
# t = load_target(targetsdir, id_to_patch[4])
# print(t)

# plt.figure()
# plt.imshow(p, interpolation='none')
# plt.show()

np.random.seed(0);

transfo_done = np.zeros((n_somemitosis,transformation_matrix.shape[0]))
transfo_ids = np.arange((transfo_done==0).sum())
np.random.shuffle(transfo_ids)
transfo_pids = (transfo_ids/transformation_matrix.shape[0]).astype('int')
transfo_tids = (transfo_ids%transformation_matrix.shape[0]).astype('int')

all_datasets = [f for f in os.listdir(patchesdir)]

n_batches = int(len(transfo_ids)/33)
print("Creating %d batches"%(n_batches))
for bid in range(n_batches):
	if( bid < 2365 ): 
		print("%d already done."%bid)
		continue
	batch = np.zeros((200, 127, 127, 3))
	batch_target = np.zeros((200,2))

	# Select with-mitosis patches
	selection_pids = transfo_pids[bid*33:(bid+1)*33]
	selection_tids = transfo_tids[bid*33:(bid+1)*33]
	# selection_patches = mitosis_patches[selection_pids]/255.
	selection_tm = transformation_matrix[selection_tids]
	for i in range(33):
		pid = selection_pids[i]
		tid = selection_tids[i]
		# batch[i] = transform(selection_patches[i], selection_tm[i])
		# batch_target[i] = mitosis_targets[i]
		batch[i] = transform(load_patch(patchesdir, id_to_patch[pid]), selection_tm[i])
		batch_target[i] = load_target(targetsdir, id_to_patch[pid])

	# Select & transform random patches
	random_selection = np.random.random((167,2))
	random_selection[:,0] *= 196*35 # patch file
	random_selection[:,1] *= 196 # patch
	random_selection = np.floor(random_selection).astype('int')
	random_transformations = get_random_transformation(167)
	for i in range(167):
		batch[33+i] = transform(np.load(os.path.join(patchesdir, all_datasets[random_selection[i,0]]))[random_selection[i,1]], random_transformations[i])
		batch_target[33+i] = np.load(os.path.join(targetsdir, all_datasets[random_selection[i,0]]))[random_selection[i,1]]
	
	# Just check if the first batch is the same...
	batch_initial = np.load('e:/data/MITOS12/train/batches/batch.0000.npy');

	np.save(os.path.join(batchesdir, "batch.%04d.npy"%bid), batch)
	np.save(os.path.join(batchesdir, "target.%04d.npy"%bid), batch_target)
	print("%d/%d"%(bid,n_batches))'''