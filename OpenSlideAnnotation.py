##
# Class for loading .ndpi images with OpenSlide, as well as .ndpa annotations.
# Converts the ndpa annotation coordinates into pixel space 
# Wrappers for getting the image as a numpy array at a certain level of magnification, and for getting the pre-computed tile coordinates (tissue-only) and the target mask
##

import xml.etree.ElementTree as etree
import openslide
import numpy as np

class OpenSlideAnnotation:
    
    def __init__(self, fname):
        self.fname = fname
        self.slide = openslide.OpenSlide(fname)     # Slide
        tree = etree.parse('%s.ndpa'%fname)    # Annotations
        self.root = tree.getroot()
        
        mppx = float(self.slide.properties['openslide.mpp-x']) #µm/px
        mppy = float(self.slide.properties['openslide.mpp-y'])
        self.nppx = mppx*1000 # nm/px
        self.nppy = mppy*1000
        self.xoff = float(self.slide.properties['hamamatsu.XOffsetFromSlideCentre']) # in nm
        self.yoff = float(self.slide.properties['hamamatsu.YOffsetFromSlideCentre'])
        self.cx,self.cy = self.slide.level_dimensions[0][0]/2, self.slide.level_dimensions[0][1]/2
    
    # Convert nanometers coordinates to pixel coordinates (in high-res image)
    def nm2xy(self,xnm,ynm):
        x = self.cx+(xnm-self.xoff)/self.nppx
        y = self.cy+(ynm-self.yoff)/self.nppy
        return int(x),int(y)
    
    # Get all annotations as pixel coordinates in the reference frame of the image at a given level of magnification
    def getAnnotationsInPixels(self,level):
        annotations = []
        for ann in self.root:
            points = ann.find('annotation').find('pointlist')
            pointsnm = []
            for p in points:
                pointsnm += [(float(p.find('x').text),float(p.find('y').text))]
            # Close the figure
            pointsnm += [pointsnm[0]]
            
            # Get all pxs in high res px space
            pointspx = [self.nm2xy(p[0],p[1]) for p in pointsnm]
            
            # Get conversion ratio to given level
            ratios = (self.slide.dimensions[0]/self.slide.level_dimensions[level][0], self.slide.dimensions[1]/self.slide.level_dimensions[level][1])
            pointspxlr = np.array([(p[0]/ratios[0],p[1]/ratios[1]) for p in pointspx])

            annotations += [pointspxlr]
        
        return annotations
    
    def get_image(self, level):
        return self.slide.read_region((0,0), level, self.slide.level_dimensions[level])
    
    def getCoords(self, level, tile_size=200, stride=8):
        return np.load('%s_%d_%d_%d.coords.npy'%(self.fname,level,tile_size,stride))
    
    def getMask(self, level):
        return np.load('%s_%d.mask.npy'%(self.fname,level))