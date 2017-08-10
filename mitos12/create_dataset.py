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

def generate_patches(fname, supervision_fname, basedir, withTargets=True):
	im = plt.imread(os.path.join(basedir, fname))
	mask = get_mask(os.path.join(basedir, supervision_fname))

	stride = 10
	print("Generating patches for %s..."%fname)
	rangex = np.arange(0,im.shape[0]-127,stride)
	rangey = np.arange(0,im.shape[1]-127,stride)
	ts = [(t//len(rangey), t%len(rangey)) for t in range(len(rangex)*len(rangey))]
	chunks = [[tx*stride+63,ty*stride+63] for tx,ty in ts] # Centers
	n_per_batch = int(np.sqrt(len(chunks)))

	for b in range(n_per_batch):
		patches = np.zeros((n_per_batch,127,127,3))
		targets = np.zeros((n_per_batch,2))
		targets[:,0] = 1.
		for i,c in enumerate(chunks[b*n_per_batch:(b+1)*n_per_batch]):
			patches[i] = im[c[0]-63:c[0]+64,c[1]-63:c[1]+64]
			if( mask[c[0],c[1]] > 0 ):
				targets[i,:] = [0., 1.]
			elif( mask[c[0]-9:c[0]+10,c[1]-9:c[1]+10].sum() > 0 ):
				targets[i,:] = [0.2, 0.8]
		np.save(os.path.join(basedir, 'patches/%s.%04d.npy'%(fname,b)), patches)
		if withTargets: np.save(os.path.join(basedir, 'targets/%s.%04d.npy'%(fname,b)), targets)

def stats_on_training_set(basedir):
	targetdir = os.path.join(basedir,'targets')
	files = [os.path.join(targetdir,f) for f in os.listdir(targetdir)]
	n_nothing = 0
	n_some = 0
	n_full = 0
	for f in files:
		t = np.load(f)
		n_nothing += (t[:,0]==1.).sum()
		n_some += (t[:,0]==0.2).sum()
		n_full += (t[:,1]==1.).sum()
	print(n_nothing,n_some,n_full)
	n_tot = n_nothing+n_some+n_full
	print(n_nothing*1./n_tot,n_some*1./n_tot,n_full*1./n_tot)

def get_sets_with_mitosis(datasetdir):
	all_datasets = [f for f in os.listdir(datasetdir)]
	with_mitosis = {}
	no_mitosis = {}
	for i,f in enumerate(all_datasets):
		data = np.load(os.path.join(datasetdir,f))
		m = data[:,1]>0
		if m.sum()>0: 
			with_mitosis[f] = m
		no_mitosis[f] = (1-m).astype('bool')
	return with_mitosis,no_mitosis

def transform(X, params):
	Y = X.copy()
	# Orientation
	if((params[1]==1) or (params[1]==3)):
		Y = Y[::-1,:,:]
	if((params[1]==2) or (params[1]==3)):
		Y = Y[:,::-1,:]
	# Noise
	Y += np.random.random(Y.shape)*params[2]-params[2]/2
	# Illumination
	Y += params[3]

	return Y

def transform_batch(X, params):
	Y = X.copy()

	# Orientation
	do_vswap = (params[:,1]==1)+(params[:,1]==3)
	do_hswap = (params[:,1]==2)+(params[:,1]==3)
	Y[do_vswap] = Y[do_vswap,::-1,:,:]
	Y[do_hswap] = Y[do_hswap,:,::-1,:]

	# Noise
	Y += np.random.random(Y.shape)*params[:,2]-params[:,2]/2

	# Illumination
	Y += params[:,3]

	return Y


def get_transformation_set_from_patch(patch,transformation_matrix):
	patches_out = []
	for t in transformation_matrix:
		patches_out += [transform(patch, t)]

	return np.array(patches_out)

def get_random_transformation(n):
	params = np.zeros((n,4))
	params[:,1] = np.floor(np.random.random((n,))*4).astype('int')
	params[:,2] = np.random.random((n,))*0.2
	params[:,3] = np.random.random((n,))*0.2 - 0.1
	return params

def create_transformation_matrix(patchesdir):
    '''
    Compute the "full transformation" matrix, stored as full_transformations_vessels.npy.
    The matrix contains the parameters necessary to get 60 patches from the original by adding 
    random transformations. The matrix has 4 columns :
    * 0     : ID of the original patch in the training_set_original_patches matrix, from which to perform the following transformations
    * 1     : Symmetries. 0 = don't touch, 1 = vertical, 2 = horizontal, 3 = both. Each original patch will produce 4 symmetry patches.
    * 2     : Noises. The matrix holds the "intensity factor" [0-0.2] of the noise to apply to the symmetry patches. 5 noisified for each symmetry patch (1 no-noise + 4 noises)
    * 3     : Illumination. 1 negative [-0.1-0], 1 neutral, 1 positive [0-0.1] illumination transformation to apply to the whole noisified patch
    '''
    n_symmetries = 4
    n_noises = 5
    n_illuminations = 3
    
    seed = 0
    np.random.seed(0)
    
    random_symmetries = np.arange(n_symmetries)
    print(random_symmetries.shape)

    random_noises = np.random.random((n_symmetries, n_noises))*0.2
    random_noises[:,0] = 0
    print(random_noises.shape)
    
    random_illuminations = np.random.random((random_noises.shape[0]*n_noises,n_illuminations))
    random_illuminations[:,0] *= -0.1
    random_illuminations[:,1] = 0.
    random_illuminations[:,2] *= 0.1
    print(random_illuminations.shape)

    full_transformations = np.zeros((random_illuminations.shape[0]*n_illuminations, 4))
    full_transformations[:,0] = np.arange(full_transformations.shape[0])
    for i in range(random_illuminations.shape[0]):
        for j in range(n_illuminations):
            full_transformations[j+i*n_illuminations,3] = random_illuminations[i,j]
    for i in range(random_noises.shape[0]):
        for j in range(n_noises):
            full_transformations[(i*n_noises+j)*n_illuminations:(i*n_noises+j+1)*n_illuminations,2] = random_noises[i,j]
    for j in range(n_symmetries):
        full_transformations[j*n_noises*n_illuminations:(j+1)*n_noises*n_illuminations,1] = random_symmetries[j]
    
    print(full_transformations.shape)
    np.save('e:/data/MITOS12/train/mitos12_full_transformations_matrix.npy', full_transformations)

def load_patch(patchesdir, ref):
	p = np.load(os.path.join(patchesdir,ref['file']))/255.

	return p[ref['id']]

def load_patches(patchesdir, f):
	return np.load(os.path.join(patchesdir,f))

def load_target(targetsdir, ref):
	p = np.load(os.path.join(targetsdir,ref['file']))

	return p[ref['id']]

def load_targets(targetsdir, f):
	return np.load(os.path.join(targetsdir,f))

# Get all files
basedir = "e:/data/MITOS12/test/"
#targetsdir = os.path.join(basedir,'targets')
patchesdir = os.path.join(basedir,'patches')
#batchesdir = os.path.join(basedir,'batches')
image_files = [f for f in os.listdir(basedir) if f.find('.bmp') > 0]
super_files = [f for f in os.listdir(basedir) if f.find('.csv') > 0]

for i in range(len(image_files)):
 	generate_patches(image_files[i], super_files[i], basedir, False)

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