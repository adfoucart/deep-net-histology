from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
from DataSource import DataSource
import csv
from random import shuffle

class MITOS12Data(DataSource):
	def __init__(self, **kwargs):
		DataSource.__init__(self)
		self.train_dirs = kwargs['train_dirs'] if 'train_dirs' in kwargs else []
		self.chunksize = kwargs['chunksize'] if 'chunksize' in kwargs else (64,64)
		self.eval_dirs = kwargs['eval_dirs'] if 'eval_dirs' in kwargs else []

		files = [ (d,os.listdir(d)) for d in self.train_dirs ]

		images = [ (Image.open(os.path.join(d,f)), os.path.join(d,f), os.path.splitext(os.path.join(d,f))[0]) for d,filesInDir in files for f in filesInDir if f.split('.')[-1] == "bmp" ]

		self.setup(images=images)

	def setup(self, **kwargs):
		self.images = kwargs['images'] if 'images' in kwargs else []
		self.initialized = True

	def getRandomImage(self):
		i = np.random.randint(len(self.images))
		return self.images[i][0]

	def getMitosisImages(self, n, **kwargs):
		assert self.initialized == True, "Data Source not initialized"

		flat = 'flat' in kwargs and kwargs['flat'] is True

		selected_image = np.random.randint(len(self.images))
		supervision = self.get_image_supervision(self.images[selected_image][2]+"_supervision.csv")
		# mitosis = self.get_supervision(self.images[selected_image][2]+".csv")
		# bboxes = self.get_boundingboxes(self.images[selected_image][2]+"_bboxes.csv")

		im = self.images[selected_image][0]
		im_control = Image.open(self.images[selected_image][2]+".jpg")
		
		chunks = []
		for s in supervision:
			if( s[2] == True ):
				chunks.append([s[0], s[1]])
		shuffle(chunks)

		if flat :
			return [np.asarray(im.crop((chunk[0], chunk[1], chunk[0]+self.chunksize[0], chunk[1]+self.chunksize[1]))).flatten().astype(np.float32)/255. for i,chunk in enumerate(chunks) if i < n]
		else:
			return [np.asarray(im.crop((chunk[0], chunk[1], chunk[0]+self.chunksize[0], chunk[1]+self.chunksize[1]))).astype(np.float32)/255. for i,chunk in enumerate(chunks) if i < n]

	def next_supervised_batch(self, n, **kwargs):
		assert self.initialized == True, "Data Source not initialized"

		flat = 'flat' in kwargs and kwargs['flat'] is True

		# Get n/2 mitosis
		m_images = self.getMitosisImages(n/2, flat=flat)
		batch = [(im,[1,0]) for im in m_images]

		selected_image = np.random.randint(len(self.images))
		chunks = np.random.random((2,n-n/2))

		im = self.images[selected_image][0]
		chunks[0,:] *= im.size[0] - self.chunksize[0]
		chunks[1,:] *= im.size[1] - self.chunksize[1]
		chunks = np.round(chunks).astype('uint16')
		if flat :
			batch += [(np.asarray(im.crop((chunk[0], chunk[1], chunk[0]+self.chunksize[0], chunk[1]+self.chunksize[1]))).flatten().astype(np.float32)/255., [0,1]) for chunk in zip(chunks[0,:],chunks[1,:])]
		else:
			batch += [(np.asarray(im.crop((chunk[0], chunk[1], chunk[0]+self.chunksize[0], chunk[1]+self.chunksize[1]))).astype(np.float32)/255., [0,1]) for chunk in zip(chunks[0,:],chunks[1,:])]
		
		shuffle(batch)
		return batch

	def next_batch(self, n, **kwargs):
		assert self.initialized == True, "Data Source not initialized"

		flat = 'flat' in kwargs and kwargs['flat'] is True

		selected_image = np.random.randint(len(self.images))
		chunks = np.random.random((2,n))

		im = self.images[selected_image][0]
		chunks[0,:] *= im.size[0] - self.chunksize[0]
		chunks[1,:] *= im.size[1] - self.chunksize[1]
		chunks = np.round(chunks).astype('uint16')
		if flat :
			return [np.asarray(im.crop((chunk[0], chunk[1], chunk[0]+self.chunksize[0], chunk[1]+self.chunksize[1]))).flatten().astype(np.float32)/255. for chunk in zip(chunks[0,:],chunks[1,:])]
		else:
			return [np.asarray(im.crop((chunk[0], chunk[1], chunk[0]+self.chunksize[0], chunk[1]+self.chunksize[1]))).astype(np.float32)/255. for chunk in zip(chunks[0,:],chunks[1,:])]
		#return [im.crop((chunk[0], chunk[1], self.chunksize[0], self.chunksize[1])) for chunk in zip(chunks[0,:],chunks[1,:])]

	def get_inputshape(self):
		return [64,64,3]

	def get_boundingboxes(self, fname):
		bboxes = []
		with open(fname, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				bbox = [int(t) for t in row]
				bboxes.append(bbox)
		return bboxes

	def get_supervision(self, fname):
		mitosis = []
		with open(fname, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				xs = [int(t) for i,t in enumerate(row) if i%2 == 0]
				ys = [int(t) for i,t in enumerate(row) if i%2 == 1]
				mitosis.append([xs,ys])
		return mitosis

	def get_image_supervision(self, fname):
		supervision = []
		with open(fname, 'rb') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				supervision.append([int(row[0]),int(row[1]),row[2]=="True"])
		return supervision

	def createFullTrainingSet(self):
		assert self.initialized == True, "Data Source not initialized"
		stride = 10

		thresh = 100

		for cur in self.images:
			im = np.array(cur[0])
			im_mitosis = np.zeros((im.shape[0], im.shape[1]))
			mitosis = self.get_supervision(cur[2]+".csv")
			for xs,ys in mitosis:
				for i in range(len(xs)):
					im_mitosis[ys[i],xs[i]] = 1
			
			rangex = np.arange(0,im.shape[0]-self.chunksize[0],stride)
			rangey = np.arange(0,im.shape[1]-self.chunksize[1],stride)
			ts = [(t/len(rangey), t%len(rangey)) for t in range(len(rangex)*len(rangey))]
			chunks = [[tx*stride,ty*stride] for tx,ty in ts]
			isMitosis = [im_mitosis[y:y+self.chunksize[1],x:x+self.chunksize[0]].sum() > 100 for x,y in chunks]

			with open(cur[2]+"_supervision.csv", "wb") as csvfile:
				print cur[2]
				writer = csv.writer(csvfile, delimiter=',')
				for i,c in enumerate(chunks):
					x = c[0]
					y = c[1]
					#pixels = list(im[y:y+self.chunksize[1],x:x+self.chunksize[0]].flatten())
					writer.writerow([x, y, isMitosis[i]])

	def get_evaluation_set(self, **kwargs):
		files = [ (d,os.listdir(d)) for d in self.eval_dirs ]
		images = [ (Image.open(os.path.join(d,f)), os.path.join(d,f), os.path.splitext(os.path.join(d,f))[0]) for d,filesInDir in files for f in filesInDir if f.split('.')[-1] == "bmp" ]
		selected_image = np.random.randint(len(images))
		stride = 10

		flat = 'flat' in kwargs and kwargs['flat'] is True

		im = images[selected_image][0]
		rangex = np.arange(0,np.array(im).shape[0]-self.chunksize[0],stride)
		rangey = np.arange(0,np.array(im).shape[1]-self.chunksize[1],stride)
		ts = [(t/len(rangey), t%len(rangey)) for t in range(len(rangex)*len(rangey))]
		chunks = [[tx*stride,ty*stride] for tx,ty in ts]

		if flat :
			return images[selected_image][2], [np.asarray(im.crop((chunk[0], chunk[1], chunk[0]+self.chunksize[0], chunk[1]+self.chunksize[1]))).flatten().astype(np.float32)/255. for chunk in chunks]
		else:
			return images[selected_image][2], [np.asarray(im.crop((chunk[0], chunk[1], chunk[0]+self.chunksize[0], chunk[1]+self.chunksize[1]))).astype(np.float32)/255. for chunk in chunks]

# basename = "/media/sf_E_DRIVE/Dropbox/ULB/Doctorat/ImageSet/MITOS12/A00_v2/A00_01"

# image = np.array(Image.open(basename+".bmp"))
# image_mitosis = np.array(Image.open(basename+".jpg"))
# image_mitosis_from_supervision = np.zeros((image.shape[0], image.shape[1]))
# with open(basename+"_supervision.csv", "rb") as csvfile:
# 	reader = csv.reader(csvfile, delimiter=",")
# 	for row in reader:
# 		if row[2] == "True":
# 			x = int(row[0])
# 			y = int(row[1])
# 			image_mitosis_from_supervision[y:y+64,x:x+64] += 1

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(image)
# plt.subplot(1,3,2)
# plt.imshow(image_mitosis)
# plt.subplot(1,3,3)
# plt.imshow(image_mitosis_from_supervision)
# plt.show()

# def getAllBoundingBoxes(dirs):
# 	files = [ (d,os.listdir(d)) for d in dirs ]
# 	csvfiles = [ os.path.join(d,f) for d,filesInDir in files for f in filesInDir if f.split('.')[-1] == "csv" ]
# 	outputfiles = [ os.path.splitext(f)[0]+"_bboxes.csv" for f in csvfiles]
# 	mitos12 = MITOS12Data()

# 	for i,csvfile in enumerate(csvfiles):
# 		coordinates = mitos12.get_supervision(csvfile)
# 		getBoundingBoxesFromCoordinates(coordinates, outputfiles[i])

# def getBoundingBoxesFromCoordinates(coordinates, saveTo):
# 	bboxes = []
# 	for c in coordinates:
# 		xs = c[0]
# 		ys = c[1]
# 		bboxes.append( (min(xs), min(ys), max(xs), max(ys)) )

# 	with open(saveTo, 'wb') as csvfile:
# 		writer = csv.writer(csvfile, delimiter=',')
# 		for bbox in bboxes:
# 			writer.writerow(bbox)
# 	print "Saved bounding boxes to :",saveTo

# 	return bboxes

#basedir = "/media/sf_E_DRIVE/Dropbox/ULB/Doctorat/ImageSet/MITOS12/"
#dirs = [os.path.join(basedir,d) for d in ["A00_v2", "A01_v2", "A02_v2", "A03_v2", "A04_v2"]]

# getAllBoundingBoxes(dirs)

# fname = os.path.join(basedir, "A00_v2/A00_01.csv")
# output = os.path.join(basedir, "A00_v2/A00_01_bboxes.csv")
#mitos12 = MITOS12Data(train_dirs=dirs)
# mitosis = mitos12.getMitosisImages(50)
# plt.figure(1)
# plt.imshow(mitosis[0])
# plt.show()

# mitosis = mitos12.get_supervision(fname)
# bboxes = getBoundingBoxesFromCoordinates(mitosis, output)
# print bboxes

# mitos12 = MITOS12Data(train_dirs=dirs)
# mitos12.createFullTrainingSet()
# mitosis = mitos12.getMitosisImages()
# print mitosis