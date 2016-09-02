from PIL import Image
import numpy as np

class Deformer:
    def __init__(self, **kwargs):
        self.nc = kwargs['nc'] if 'nc' in kwargs else None
        self.rot = kwargs['rot'] if 'rot' in kwargs else False
        self.flip = kwargs['flip'] if 'flip' in kwargs else False
        self.resizeTo = kwargs['resizeTo'] if 'resizeTo' in kwargs else None

    def addNoise(self, im):
        return im + np.random.rand(im.size[0], im.size[1], 3)*self.nc*255 if self.nc != None else np.asarray(im)

    def addRot(self, im):
        if self.rot == False: return im

        angle = int(np.random.rand()*4)

        if angle==0: return im
        if angle==1: return im.transpose(Image.ROTATE_90)
        if angle==2: return im.transpose(Image.ROTATE_180)
        if angle==3: return im.transpose(Image.ROTATE_270)

        raise ValueError('Invalid rotation value ',angle)

    def addFlip(self, im):
        if self.flip == False: return im

        direction = int(np.random.rand()*3)

        if direction == 0: return im
        if direction == 1: return im.transpose(Image.FLIP_LEFT_RIGHT)
        if direction == 2: return im.transpose(Image.FLIP_TOP_BOTTOM)

        raise ValueError('Invalid flip value ',direction)

    def addResize(self, im):
        return im.resize(self.resizeTo)

    def apply(self, im):
        return self.addNoise(self.addResize(self.addRot(self.addFlip(im))))