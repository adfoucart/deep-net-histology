import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from matplotlib import pyplot as plt

def doStretch(h, p2, p98, p2p=0.6, p98p=0.99):
	hp = h.copy()
	maskLow = h<p2
	maskMid = (h>=p2)*(h<p98)
	maskHigh = h>=p98
	hp[maskLow] = h[maskLow]*p2p/p2
	hp[maskMid] = p2p+((h[maskMid]-p2)/(p98-p2))*(p98p-p2p)
	hp[maskHigh] = p98p+((h[maskHigh]-p98)/(1-p98))*(1-p98p)
	return hp

def stretchImageHue(imrgb):
	# Image must be stored as 0-1 bound float. If it's 0-255 int, convert
	if( imrgb.max() > 1 ):
		imrgb = imrgb*1./255

	# Transform to HSV
	imhsv = rgb2hsv(imrgb)

	# Find 2-98 percentiles of H histogram (except de-saturated pixels)
	plt.figure()
	plt.hist(imhsv[imhsv[:,:,1]>0.1,0].flatten(), bins=360)
	p2, p98 = np.percentile(imhsv[imhsv[:,:,1]>0.1,0], (2, 98))
	print p2, p98

	imhsv[:,:,0] = doStretch(imhsv[:,:,0], p2, p98, 0.6, 0.99)
	plt.figure()
	plt.hist(imhsv[imhsv[:,:,1]>0.1,0].flatten(), bins=360)	

	imrgb_stretched = hsv2rgb(imhsv)
	plt.figure()
	plt.imshow(imrgb)
	plt.figure()
	plt.imshow(imrgb_stretched)

	plt.show()

im = plt.imread("../../ImageSet/MITOS12/A03_v2/A03_02.bmp")
stretchImageHue(im)