
# coding: utf-8

import pandas as pd
pd.__version__
from skimage.io import imread,imsave
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
import csv
import sys

def export_files(cells_only = False):
    super_df = pd.read_pickle('super_df.pkl')
    scale = .454 #micron per pixel
    scale2= .453*.455 #micron per pixel2
    super_df['area'] = scale2 * super_df['area']
    super_df['convex_area'] = scale2 * super_df['convex_area'] 
    super_df['filled_area'] = scale2 * super_df['filled_area'] 
    super_df['perimeter'] = scale * super_df['perimeter'] 
    super_df['major_axis_length'] = scale * super_df['major_axis_length'] 
    super_df['minor_axis_length'] = scale * super_df['minor_axis_length'] 
    super_df['equivalent_diameter'] = scale * super_df['equivalent_diameter'] 

    #import vessels data
    '''dfv = pd.read_pickle('all_DF_vessels.pkl')
    print dfv.columns
    dfv.head()'''


    # Have a look at bounding boxes size to see if 256x256 is a good sliding window size
    '''xs = (np.asarray(super_df[super_df['valid']!=3][['bb_x1']])-np.asarray(super_df[super_df['valid']!=3][['bb_x0']]))[:,0]
    print xs.min(),xs.max(),np.median(xs)
    plt.hist(xs)
    plt.show()
    ys = (np.asarray(super_df[super_df['valid']!=3][['bb_y1']])-np.asarray(super_df[super_df['valid']!=3][['bb_y0']]))[:,0]
    print ys.min(),ys.max(),np.median(ys)
    plt.hist(ys)'''

    # Check the number of examples for different types of cells 
    for t in range(4):
        print t, super_df[super_df['valid']==t]['valid'].count()

    d = [f for f in os.listdir('.') if (f[-4:] == '.tif' and f.find('rgb_tile')==0)]
    if cells_only:
        for f in d: export_file_cells_only(super_df, f)
    else:
        for f in d : export_file(super_df, f)

def export_file_cells_only(super_df, fname, tile_size=128):
    print "exporting file: %s"%fname
    ii = fname.find('_i')
    jj = fname.find('_j')
    i = int(fname[ii+2:ii+4])
    j = int(fname[jj+2:jj+4])

    # Get all cells from tile
    in_tile = super_df[super_df['i']==i][super_df['j']==j]
    cells = np.asarray(in_tile[['x','y','valid','convex_area']])

    if len(cells) <= 0:
        print "nothing here... skipping."
        return

    # Get tile
    im = plt.imread(fname)

    print "Creating and saving dataset to %s_%d_cells_only.npy..."%(fname,tile_size)
    dataset = []
    for c in cells:
        box = [int(c[1])-tile_size/2,int(c[1])+tile_size/2, int(c[0])-tile_size/2, int(c[0])+tile_size/2]
        if box[0] < 0 or box[1] >= im.shape[0] or box[2] < 0 or box[3] >= im.shape[1]:
            continue
        p = [0.,0.,0.,0.]
        p[int(c[2])] = 1.
        l = box + p
        dataset += [l]

    dataset = np.array(dataset)
    np.save('dataset/%s_%d_cells_only.npy'%(fname,tile_size), dataset)

def export_file(super_df, fname, tile_size=128, stride=16):
    print "exporting file: %s"%fname
    ii = fname.find('_i')
    jj = fname.find('_j')
    i = int(fname[ii+2:ii+4])
    j = int(fname[jj+2:jj+4])

    # Get all cells from tile
    in_tile = super_df[super_df['i']==i][super_df['j']==j]
    cells = np.asarray(in_tile[['x','y','valid','convex_area']])

    # Get tile
    im = plt.imread(fname)

    # Show tile with all cells
    '''plt.figure(figsize=(15,15))
    plt.imshow(im)
    plt.plot(cells[cells[:,2]==0,0],cells[cells[:,2]==0,1], 'b+')
    plt.plot(cells[cells[:,2]==1,0],cells[cells[:,2]==1,1], 'k+')
    plt.plot(cells[cells[:,2]==2,0],cells[cells[:,2]==2,1], 'r+')
    plt.xlim([0,im.shape[1]])
    plt.ylim([im.shape[0],0])
    plt.axis('off')
    plt.show()'''

    # Compute probabilities map
    im_probs = np.zeros((im.shape[0], im.shape[1], 4)).astype('float')

    #im_probs[:,:,3] = 1.
    # print im_probs.shape
    for c in cells:
        #t = int(c[3]**(0.5))
        im_probs[int(c[1]),int(c[0]),int(c[2])] = int(c[3])
        #if c[2]==0: im_probs[int(c[1])-t:int(c[1])+t,int(c[0])-t:int(c[0])+t,0] += 1.
        #if c[2]==1: im_probs[int(c[1])-t:int(c[1])+t,int(c[0])-t:int(c[0])+t,1] += 1.
        #if c[2]==2: im_probs[int(c[1])-t:int(c[1])+t,int(c[0])-t:int(c[0])+t,2] += 1.

    # im_probs[:,:,3] = (1.-(im_probs.max(axis=2)>0))
    # im_probs[im_probs>1.] = 1.


    # Show probability map
    '''plt.figure(figsize=(15,15))
    plt.imshow(im_probs[:,:,:3], interpolation='none')
    plt.plot(cells[cells[:,2]==0,0],cells[cells[:,2]==0,1], 'b+')
    plt.plot(cells[cells[:,2]==1,0],cells[cells[:,2]==1,1], 'k+')
    plt.plot(cells[cells[:,2]==2,0],cells[cells[:,2]==2,1], 'r+')
    plt.xlim([0,im.shape[1]])
    plt.ylim([im.shape[0],0])
    # plt.figure(figsize=(15,15))
    # plt.imshow(im_probs[:,:,3])
    plt.show()'''

    # Create and export dataset
    tile_area = tile_size**2

    x,y = np.ogrid[-tile_size/2:tile_size/2,-tile_size/2:tile_size/2]
    cir = (x**2+y**2)
    cir = (cir.max()-cir)*1./cir.max()
    cir[cir<0.5] = 0.5
    mask = np.zeros((tile_size,tile_size,4))
    mask[:,:,0] = cir
    mask[:,:,1] = cir
    mask[:,:,2] = cir
    mask[:,:,3] = cir

    range0 = np.arange(0,im.shape[0]-tile_size,stride)
    range1 = np.arange(0,im.shape[1]-tile_size,stride)
    ts = [(t/len(range1), t%len(range1)) for t in range(len(range0)*len(range1))]
    chunks = [[t0*stride,t1*stride] for t0,t1 in ts]

    print "Creating and saving dataset to %s_%d.npy..."%(fname,tile_size)
    dataset = []
    i=0
    max_per_file = 10000
    # with open('%s_%d.csv'%(fname,tile_size), 'wb') as f:
        # writer = csv.writer(f)
    for c in chunks:
        # tile = im[c[0]:c[0]+tile_size,c[1]:c[1]+tile_size,:]
        probs = im_probs[c[0]:c[0]+tile_size,c[1]:c[1]+tile_size,:]

        if probs[:,:,:3].max() > 1.:
            probs = (probs*mask).sum(axis=0).sum(axis=0)/tile_area
            probs /= probs.sum()
            l = [c[0],c[0]+tile_size,c[1],c[1]+tile_size]+list(probs)
            dataset += [l]
            # writer.writerow(l)
            i+=1
    if i > max_per_file:
        # Split if it's too big
        dataset = np.array(dataset)
        cur = 0
        while(cur < dataset.shape[0]):
            n = min(max_per_file, dataset.shape[0]-cur)
            savename = 'dataset/%s_%d_%d.npy'%(fname,tile_size, cur)
            np.save(savename, dataset[cur:cur+n])
            print "saved file %s"%savename
            cur += n
    elif i > 0:
        dataset = np.array(dataset)
        np.save('dataset/%s_%d.npy'%(fname,tile_size), dataset)
    else:
        print "nothing here... skipping."

    #print i,len(chunks)
    #dataset = np.zeros((i,(tile_area+4)))
    #print dataset.shape

    '''with open("%s_128.csv"%(fname), "wb") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for c in chunks:
            tile = im[c[0]:c[0]+tile_size,c[1]:c[1]+tile_size,:]
            probs = im_probs[c[0]:c[0]+tile_size,c[1]:c[1]+tile_size,:].sum(axis=0).sum(axis=0)/tile_area
            if probs[3] != 1.:
                l = list(tile.flatten())+list(probs)
                writer.writerow(l)'''

def export_tiles(fname, savedir, datasetdir='dataset', tile_size=128):
    # Get all the dataset files for this tile
    datasets = [f for f in os.listdir(datasetdir) if f.find('%s_%d'%(fname,tile_size)) == 0]
    print datasets
    for f in datasets:
        dataset = np.load(os.path.join(datasetdir, '%s'%f))

        im = plt.imread(fname)
        tiles = np.zeros((dataset.shape[0], tile_size, tile_size, 3)).astype('int')
        for i,d in enumerate(dataset):
            tiles[i,:,:,:] = im[d[0]:d[1],d[2]:d[3]]
        savename = os.path.join(savedir, '%s_patches.npy'%(f[:-4]))
        np.save(savename, tiles)
        print "saved file %s"%f

    # dataset = np.load('dataset/%s_%d.npy'%(fname,tile_size)).astype('int')

    # # Get image
    # im = plt.imread(fname)

    # if dataset.shape[0] > 15000:
    #     # Split if it's too big
    #     cur = 0
    #     while(cur < dataset.shape[0]):
    #         n = min(15000, dataset.shape[0]-cur)
    #         tiles = np.zeros((n, tile_size, tile_size, 3)).astype('int')
    #         for i,d in enumerate(dataset[cur:cur+n]):
    #             tiles[i,:,:,:] = im[d[0]:d[1],d[2]:d[3]]
    #         savename = os.path.join(savedir, '%s_%d_patches_%d.npy'%(fname,tile_size, cur))
    #         print tiles.shape
    #         np.save(savename, tiles)
    #         print "saved file %s"%savename
    #         cur += n
    # else:
    #     # Get tiles
    #     tiles = np.zeros((dataset.shape[0], tile_size, tile_size, 3)).astype('int')

    #     for i,d in enumerate(dataset):
    #         tiles[i,:,:,:] = im[d[0]:d[1],d[2]:d[3]]
        
    #     savename = os.path.join(savedir, '%s_%d_patches.npy'%(fname,tile_size))
    #     np.save(savename, tiles)
    #     print "saved file %s"%fname

def make_equal_batches(data_dir, patches_dir, out_dir):
    files = [f for f in os.listdir(data_dir) if f.find('.npy') >= 0]

    n_per_class = [70,30,20,80]
    cur_per_class = [0,0,0,0]
    patches_by_class = [None,None,None,None]
    batch_n = 0
    for fname in files:
        print fname
        saved_something = False
        data = np.load(os.path.join(data_dir,fname))
        patches = np.load(os.path.join(patches_dir, '%s_patches.npy'%fname[:-4]))
        target = data[:,-4:].argmax(axis=1)

        for i in range(4):
            if patches_by_class[i] is None:
                patches_by_class[i] = patches[target==i]
            else:
                patches_by_class[i] = np.vstack([patches_by_class[i], patches[target==i]])

        while np.array([(patches_by_class[i].shape[0]-n_per_class[i])>0 for i in range(4)]).sum() == 4:
            batch = np.zeros((200,128,128,3))
            batch_target = np.zeros((200,4))
            cur_in_batch = 0
            for i,n in enumerate(n_per_class):
                batch[cur_in_batch:cur_in_batch+n,:,:,:3] = patches_by_class[i][:n,:,:,:]
                batch_target[cur_in_batch:cur_in_batch+n,i] = 1.
                cur_in_batch += n
                patches_by_class[i] = patches_by_class[i][n:,:,:,:]
            np.save(os.path.join(out_dir, 'batch_%05d_patch.npy'%batch_n), batch)
            np.save(os.path.join(out_dir, 'batch_%05d_target.npy'%batch_n), batch_target)
            batch_n += 1
            saved_something = True

        if saved_something: patches_by_class = [None,None,None,None]

def create_vessel_dataset(datasetdir, patchesdir):
    super_df = pd.read_pickle('super_df.pkl')
    vessel_cells = super_df[super_df['vessel_id']>0]

    images = np.array(vessel_cells[['i','j']].drop_duplicates())

    tiles_dict = {}
    files = [f for f in os.listdir('.') if (f[-4:] == '.tif' and f.find('rgb_tile')==0)]
    for fname in files:
        ii = fname.find('_i')
        jj = fname.find('_j')
        i = int(fname[ii+2:ii+4])
        j = int(fname[jj+2:jj+4])
        tiles_dict["%d.%d"%(i,j)] = fname
    
    # The dataset takes patches of 165x165, so that we can use random translations to get more 128x128 patches
    for tile_coord in images:
        vessels_tile = vessel_cells[vessel_cells['i']==tile_coord[0]][vessel_cells['j']==tile_coord[1]]
        vessel_ids = set(vessels_tile['vessel_id'].get_values())
        patches = np.zeros((len(vessel_ids),165,165,3)).astype('uint8')
        fname = tiles_dict["%d.%d"%(tile_coord[0],tile_coord[1])]
        im = plt.imread(fname)
        ids_to_keep = np.ones((len(vessel_ids),)).astype('bool')
        for i,v in enumerate(vessel_ids):
            v_cells = vessels_tile[vessels_tile['vessel_id']==v][['x', 'y', 'vessel_id', 'i', 'j']]
            xy_cells = np.array(v_cells[['x','y','i','j']])
            centroid = xy_cells[:,0:2].mean(axis=0)
            std = xy_cells[:,0:2].std(axis=0)
            if( centroid[0] < 82 or centroid[0] >= im.shape[1]-82 or centroid[1] < 82 or centroid[1] >= im.shape[0]-82 ):
                ids_to_keep[i] = 0
            else:
                patches[i] = im[centroid[1]-82:centroid[1]+83,centroid[0]-82:centroid[0]+83,:]
        np.save(os.path.join(patchesdir, "%s.vessels.npy"%fname), patches[ids_to_keep])


def create_nonvessel_dataset(datasetdir, patchesdir):
    super_df = pd.read_pickle('super_df.pkl')
    nonvessel_cells = super_df[super_df['vessel_id']<=0]

    images = np.array(nonvessel_cells[['i','j']].drop_duplicates())

    tiles_dict = {}
    files = [f for f in os.listdir('.') if (f[-4:] == '.tif' and f.find('rgb_tile')==0)]
    for fname in files:
        ii = fname.find('_i')
        jj = fname.find('_j')
        i = int(fname[ii+2:ii+4])
        j = int(fname[jj+2:jj+4])
        tiles_dict["%d.%d"%(i,j)] = fname
    
    # The dataset takes patches of 165x165, so that we can use random translations to get more 128x128 patches
    for tile_coord in images:
        if "%d.%d"%(tile_coord[0],tile_coord[1]) not in tiles_dict: 
            print "Skipping %d.%d"%(tile_coord[0],tile_coord[1])
            continue

        fname = tiles_dict["%d.%d"%(tile_coord[0],tile_coord[1])]
        
        cells_coord = np.array(nonvessel_cells[nonvessel_cells['i']==tile_coord[0]][nonvessel_cells['j']==tile_coord[1]][['x', 'y']])

        im = plt.imread(fname)

        ids_to_keep = (cells_coord[:,0] >= 82)*(cells_coord[:,0] < im.shape[1]-82)*(cells_coord[:,1] >= 82)*(cells_coord[:,1] < im.shape[0]-82)
        cells_coord = cells_coord[ids_to_keep]
        print cells_coord.shape
        for i in range(1+cells_coord.shape[0]/1000):
            coords = cells_coord[i*1000:min((i+1)*1000,cells_coord.shape[0])]
            patches = np.zeros((coords.shape[0],165,165,3))
            for j,coord in enumerate(coords):
                patches[j] = im[coord[1]-82:coord[1]+83,coord[0]-82:coord[0]+83,:]
            break
            # np.save(os.path.join(patchesdir, "%s.nonvessels.%d.npy"%(fname,i)), patches)
        break
        print "[Done] %s"%fname

# def random_symmetry(self, X):
#         do_v_sym_on = np.random.random((X.shape[0],))>0.5
#         do_h_sym_on = np.random.random((X.shape[0],))>0.5
#         X[do_v_sym_on] = X[do_v_sym_on,::-1,:,:]
#         X[do_h_sym_on] = X[do_h_sym_on,:,::-1,:]

def create_transformation_matrix(datasetdir, patchesdir):
    vessels = [os.path.join(patchesdir,f) for f in os.listdir(patchesdir) if f.find('.vessels.') > 0]

    '''
    Compute the "full transformation" matrix, stored as full_transformations_vessels.npy.
    The matrix contains the parameters necessary to get 159.750 patches from the original 355 by adding 
    random transformations. The matrix has 6 columns :
    * 0     : ID of the original patch in the training_set_original_patches matrix, from which to perform the following transformations
    * 1-2   : Tx,Ty for the random translation within the 165x165 patch, the coordinates from which to get the final 128x128 patch. (10 random translations are taken for each original patch)
    * 3     : Symmetries. 0 = don't touch, 1 = vertical, 2 = horizontal. Each translated patch will produce 3 symmetry patches.
    * 4     : Noises. The matrix holds the "intensity factor" [0-0.2] of the noise to apply to the symmetry patches. 5 noisified for each symmetry patch (1 no-noise + 4 noises)
    * 5     : Illumination. 1 negative [-0.1-0], 1 neutral, 1 positive [0-0.1] illumination transformation to apply to the whole noisified patch
    '''
    n_translations = 10
    n_symmetries = 3
    n_noises = 5
    n_illuminations = 3
    
    seed = 0
    np.random.seed(0)
    random_vessel_files = np.arange(len(vessels))
    np.random.shuffle(random_vessel_files)

    # Keep 30 vessels for tests
    training_set = random_vessel_files[:65]
    test_set = random_vessel_files[65:]

    training_set_original_patches = np.vstack([np.load(vessels[vf]) for vf in training_set])
    np.save('vessels_training_set_original_patches.npy', training_set_original_patches)

    test_set_patches = np.vstack([np.load(vessels[vf]) for vf in test_set])
    np.save('vessels_test_set_patches.npy', test_set_patches)

    n_original = training_set_original_patches.shape[0]

    random_translations = np.floor(np.random.random((n_original,n_translations,2))*38).astype('uint8')
    print random_translations.shape

    random_symmetries = np.vstack([np.arange(n_symmetries) for t in range(n_original*n_translations)])
    print random_symmetries.shape

    random_noises = np.random.random((random_symmetries.shape[0]*n_symmetries, n_noises))*0.2
    random_noises[:,0] = 0
    print random_noises.shape
    
    random_illuminations = np.random.random((random_noises.shape[0]*n_noises,n_illuminations))
    random_illuminations[:,0] *= -0.1
    random_illuminations[:,1] = 0.
    random_illuminations[:,2] *= 0.1
    print random_illuminations.shape

    full_transformations = np.zeros((random_illuminations.shape[0]*random_illuminations.shape[1], 6))
    for i in range(random_illuminations.shape[0]):
        for j in range(n_illuminations):
            full_transformations[j+i*n_illuminations,5] = random_illuminations[i,j]
    for i in range(random_noises.shape[0]):
        for j in range(n_noises):
            full_transformations[(i*n_noises+j)*n_illuminations:(i*n_noises+j+1)*n_illuminations,4] = random_noises[i,j]
    for i in range(random_symmetries.shape[0]):
        for j in range(n_symmetries):
            full_transformations[(i*n_symmetries+j)*n_noises*n_illuminations:(i*n_symmetries+j+1)*n_noises*n_illuminations,3] = random_symmetries[i,j]
    for i in range(random_translations.shape[0]):
        for j in range(n_translations):
            full_transformations[(i*n_translations+j)*n_symmetries*n_noises*n_illuminations:(i*n_translations+j+1)*n_symmetries*n_noises*n_illuminations,1] = random_translations[i,j,0]
            full_transformations[(i*n_translations+j)*n_symmetries*n_noises*n_illuminations:(i*n_translations+j+1)*n_symmetries*n_noises*n_illuminations,2] = random_translations[i,j,1]
        full_transformations[i*n_translations*n_symmetries*n_noises*n_illuminations:(i+1)*n_translations*n_symmetries*n_noises*n_illuminations,0] = i # ID OF ORIGINAL PATCH
    

    print full_transformations.shape
    np.save('vessels_full_transformations_matrix.npy', full_transformations)

def transform(X, params):
    # Translation
    Y = X[params[0]:params[0]+128,params[1]:params[1]+128,:]
    # Orientation
    if params[2]==1:
        Y = Y[::-1,:,:]
    if params[2]==2:
        Y = Y[:,::-1,:]
    # Noise
    Y += np.random.random(Y.shape)*params[3]-params[3]/2
    # Illumination
    Y += params[4]

    return Y

def export_vessel_dataset(datasetdir, patchesdir):
    '''
    The "full transformations" matrix contains all the information to produce the output patches, including the reference id to the original patch.
    We randomize the batches so that they contain data created from different original images
    '''
    training_set_original_patches = np.load("vessels_training_set_original_patches.npy")*1./255
    full_transformations = np.load("vessels_full_transformations_matrix.npy")

    randomized_ids = np.arange(full_transformations.shape[0])
    np.random.shuffle(randomized_ids)

    batch_size = 60
    N = (len(randomized_ids)/60)*60
    batch = np.zeros((batch_size,128,128,3))

    print "Creating batches..."
    for i,pid in enumerate(randomized_ids):
        sys.stdout.write("\r[%.2f%]"%(i*1./randomized_ids))
        sys.stdout.flush()
        T = full_transformations[pid]
        batch[i%batch_size] = transform(training_set_original_patches[T[0]], T[1:])
        if i%batch_size == (batch_size-1):
            np.save(os.path.join(patchesdir,'batch_60vessels_%04d.npy'%(i/batch_size)), batch)
    sys.stdout.write("[Done]")
    sys.stdout.flush()

'''
def export_vessel_testset(datasetdir, patchesdir):
    test_set_patches = np.load("vessels_test_set_patches.npy")*1./255
    print test_set_patches.shape
'''

def export_nonvessel_dataset(datasetdir, patchesdir):
    nonvessels = [os.path.join(patchesdir,f) for f in os.listdir(patchesdir) if f.find('.nonvessels.') > 0]

    np.random.seed(0)
    nv_ids = np.arange(len(nonvessels))
    np.random.shuffle(nv_ids)

    training = nv_ids[:336]
    test = nv_ids[336:]

    '''
    We have a ton of examples for the nonvessel dataset.
    To create the batch, we randomly select 7 files, and combine groups of 20 patches from these files to produces 140-image batches, until one of the file runs out of patches. 
    We then move out to the next group. 
    '''
    t = 0
    for i in range(48):
        print "%d/48"%i
        batches = [np.load(nonvessels[f])*1./255 for f in training[i*7:(i+1)*7]]
        b_ids = [np.arange(b.shape[0]) for b in batches]
        for b in b_ids: np.random.shuffle(b)
        n_batches = np.min([len(b)/20 for b in b_ids])
        transform_params = np.random.random((n_batches*20,5))
        transform_params[:,:2] = np.floor(transform_params[:,:2]*38)
        transform_params[:,2] = np.floor(transform_params[:,2]*3)
        transform_params[:,3] = transform_params[:,3]*0.2
        transform_params[:,4] = (transform_params[:,4]*0.2)-0.1
        for j in range(n_batches):
            nv_batch = np.vstack([b[b_ids[bi][j*20:(j+1)*20]] for bi,b in enumerate(batches)])
            nvt_batch = np.zeros((140,128,128,3))
            for k in range(140):
                nvt_batch[k] = transform(nv_batch[k], transform_params[j*20+k])
            np.save(os.path.join(patchesdir,'batch_140nonvessels_%04d.npy'%t), nvt_batch)
            t += 1
    print "Done"

    # batch_size = 60
    # N = (len(randomized_ids)/60)*60
    # batch = np.zeros((batch_size,128,128,3))

    # print "Creating batches..."
    # for i,pid in enumerate(randomized_ids):
    #     sys.stdout.write("\r[%.2f%]"%(i*1./randomized_ids))
    #     sys.stdout.flush()
    #     T = full_transformations[pid]
    #     batch[i%batch_size] = transform(training_set_original_patches[T[0]], T[1:])
    #     if i%batch_size == (batch_size-1):
    #         np.save(os.path.join(patchesdir,'batch_60vessels_%04d.npy'%(i/batch_size)), batch)
    # sys.stdout.write("[Done]")
    # sys.stdout.flush()


import os

# export_files(True)

# Tile file -- this should be adapted to cover all tiles
# fname = "rgb_tile_155_i18_j05.tif"
# export_tiles(fname)
# export_file(fname)

# datasetdir = 'e:/data/algae_dataset_cells_only'
# patchesdir = 'e:/data/algae_patches_cells_only'
# d = [f[:f.find('.tif')+4] for f in os.listdir(datasetdir)]
# print d
# for f in d: export_tiles(f, patchesdir, datasetdir)

# d = [f[:f.find('.tif')+4] for f in os.listdir('dataset')]
# d = [f for f in os.listdir('.') if (f[-4:] == '.tif' and f.find('rgb_tile')==0)]
# for f in d: export_tiles(f, 'e:/data/algae_patches')
# f = "rgb_tile_056_i16_j11.tif"
# export_tiles(f, '/mnt/e/data/algae_patches')

#out_dir = 'e:/data/algae_dataset_equal_batches'
#make_equal_batches(datasetdir, patchesdir, out_dir)

datasetdir = 'e:/data/algae_dataset_vessels'
patchesdir = 'e:/data/algae_patches_vessels'
# create_vessel_dataset(datasetdir, patchesdir)
create_nonvessel_dataset(datasetdir, patchesdir)
# create_transformation_matrix(datasetdir, patchesdir)
# export_vessel_dataset(datasetdir, patchesdir)
# export_vessel_testset(datasetdir,patchesdir)
# export_nonvessel_dataset(datasetdir, patchesdir)