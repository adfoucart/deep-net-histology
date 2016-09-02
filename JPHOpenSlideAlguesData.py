from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from DataSource import DataSource
import csv
from random import shuffle
from openslide import OpenSlide, OpenSlideError
from skimage import color, feature

class JPHOpenSlideAlguesData(DataSource):
    def __init__(self, **kwargs):
        DataSource.__init__(self)
        self.train_dirs = kwargs['train_dirs'] if 'train_dirs' in kwargs else []
        self.chunksize = kwargs['chunksize'] if 'chunksize' in kwargs else (64,64)
        
        files = [ (d,os.listdir(d)) for d in self.train_dirs ]

        images = [ (os.path.join(d,f), os.path.splitext(os.path.join(d,f))[0]) for d,filesInDir in files for f in filesInDir if f.split('.')[-1] == "ndpi" and f.find("HE") > 0 ]

        self.setup(images=images)

    def setup(self, **kwargs):
        print "Preparing dataset...",
        self.images = kwargs['images'] if 'images' in kwargs else []

        self.masks = [ self.get_mask(OpenSlide(im[0])) for im in self.images ]

        self.initialized = True
        print "Done."

    def get_mask(self, osr):
        levels = osr.level_dimensions

        lowest_res = osr.read_region((0,0), len(levels)-2, levels[len(levels)-2])
        lowest_res_np = np.array(lowest_res)[:,:,:3]
        lr_gs = color.rgb2gray(lowest_res_np)
        lr_hog = feature.hog(lr_gs, 4, (16,16), (1,1), visualise=False)
        n_cells = lr_hog.shape[0]/4
        shape_hog = (lr_gs.shape[0]/16, lr_gs.shape[1]/16)
        mask = np.zeros((n_cells,))
        for c in range(n_cells):
            h = lr_hog[c*4:(c+1)*4].sum() > 0.999
            mask[c] = h

        return mask.reshape(shape_hog).astype('bool')


    def next_batch(self, n, **kwargs):
        assert self.initialized == True, "Data Source not initialized"

        flat = 'flat' in kwargs and kwargs['flat'] is True
        noise = 'noise' in kwargs and kwargs['noise'] is True
        nc = kwargs['nc'] if 'nc' in kwargs else 0.05
        level = kwargs['level'] if 'level' in kwargs else 0
        isfloat = kwargs['isfloat'] if 'isfloat' in kwargs else False

        selected_image = np.random.randint(len(self.images))
        chunks = np.random.random((3,n))

        im_path = self.images[selected_image][0]
        osr = OpenSlide(im_path)
        mask = self.masks[selected_image]
        imsize = osr.level_dimensions[level]
        ratios = [imsize[0]*1./mask.shape[0], imsize[1]*1./mask.shape[1]]

        n_px_in_mask = mask.sum()
        indexes = np.arange(n_px_in_mask)+1
        mask_indexes = np.zeros(mask.shape)
        mask_indexes[mask] = indexes

        chunks[0,:] *= n_px_in_mask # index in mask
        chunks[1,:] *= (ratios[0]-self.chunksize[0])    # y pixel offset in higher def image from top-left of region
        chunks[2,:] *= (ratios[1]-self.chunksize[1])    # x pixel offset in higher def image from top-left of region
        if isfloat:
            chunks = chunks.astype('float')
            if chunks.max() > 1:
                chunks = chunks / 255.
        else:
            if chunks.max() < 1:
                chunks = chunks*255
            chunks = np.round(chunks).astype('uint16')

        lowres_coordinates = [np.where(mask_indexes==c) for c in chunks[0,:]]
        highres_coordinates = [ ( int((lr[0]*ratios[0] + chunks[1,i])[0]), int((lr[1]*ratios[1] + chunks[2,i])[0]) ) for i,lr in enumerate(lowres_coordinates) ]

        if noise:
            if flat:
                toAdd = np.random.rand(n, self.chunksize[0]*self.chunksize[1]*3)*nc
            else:
                toAdd = np.random.rand(n, self.chunksize[0], self.chunksize[1], 3)*nc
        else:
            if flat:
                toAdd = np.zeros((n, self.chunksize[0]*self.chunksize[1]*3))
            else:
                toAdd = np.zeros((n, self.chunksize[0], self.chunksize[1], 3))

        if flat :
            return [np.asarray(osr.read_region((chunk[0], chunk[1]), level, self.chunksize)).flatten().astype(np.float32)[:,:,:3]/255.+toAdd[i] for i,chunk in enumerate(highres_coordinates)]
        else:
            return [np.asarray(osr.read_region((chunk[0], chunk[1]), level, self.chunksize)).astype(np.float32)[:,:,:3]/255.+toAdd[i] for i,chunk in enumerate(highres_coordinates)]

    def get_inputshape(self):
        return [self.chunksize[0],self.chunksize[1],3]
