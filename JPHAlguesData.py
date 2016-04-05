from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from DataSource import DataSource

class JPHAlguesData(DataSource):
    def __init__(self, **kwargs):
        DataSource.__init__(self)
        self.train_dirs = kwargs['train_dirs'] if 'train_dirs' in kwargs else []

        files = [ (d,os.listdir(d)) for d in self.train_dirs ]
        
        self.setup(paths=[os.path.join(d,f) for d,filesInDir in files for f in filesInDir if f.split('.')[-1] == "png" ])

        # images = [ (Image.open(os.path.join(d,f)), os.path.join(d,f)) for d,filesInDir in files for f in filesInDir if f.split('.')[-1] == "png" ]

        # self.setup(images=images)

    def setup(self, **kwargs):
        self.paths = kwargs['paths'] if 'paths' in kwargs else []
        self.initialized = True

    def next_batch(self, n, **kwargs):
        assert self.initialized == True, "Data Source not initialized"

        flat = 'flat' in kwargs and kwargs['flat'] is True

        selected_image = np.random.randint(len(self.paths))
        chunks = np.random.random((2,n))

        im = Image.open(self.paths[selected_image])
        chunksize = (128,128)
        chunks[0,:] *= im.size[0] - chunksize[0]
        chunks[1,:] *= im.size[1] - chunksize[1]
        chunks = np.round(chunks).astype('uint16')
        if flat :
            return [np.asarray(im.crop((chunk[0], chunk[1], chunk[0]+chunksize[0], chunk[1]+chunksize[1]))).flatten().astype(np.float32)/255. for chunk in zip(chunks[0,:],chunks[1,:])]
        else:
            return [np.asarray(im.crop((chunk[0], chunk[1], chunk[0]+chunksize[0], chunk[1]+chunksize[1]))).astype(np.float32)/255. for chunk in zip(chunks[0,:],chunks[1,:])]
        #return [im.crop((chunk[0], chunk[1], chunksize[0], chunksize[1])) for chunk in zip(chunks[0,:],chunks[1,:])]

    def get_inputshape(self):
        return [128,128,3]