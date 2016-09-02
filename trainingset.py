import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from MITOS12Data import MITOS12Data

def get_tensor_from_jpgs(paths, outsize=None):
    queue = tf.train.string_input_producer(paths)
    reader = tf.WholeFileReader()

    images = []
    for i in range(len(paths)):
        key,value = reader.read(queue)
        images.append(tf.image.decode_jpeg(value))

    if outsize != None:
        return tf.image.resize_images(images, outsize[0], outsize[1], tf.ResizeMethod.BILINEAR)
    else:
        return images

basedir = "/media/sf_E_DRIVE/Dropbox/ULB/Doctorat/ImageSet/MITOS12/"
mitos12 = MITOS12Data(train_dirs=[os.path.join(basedir,d) for d in ["A00_v2", "A01_v2", "A02_v2", "A03_v2", "A04_v2"]], chunksize=(128,128))

batch = mitos12.next_batch(50)
plt.imshow(batch[0])
plt.show()
# npImages = [np.array(im[0]) for im in mitos12.images]

# images = tf.placeholder(tf.uint8, [None, 2084, 2084, 3])
# patches = tf.extract_image_patches( images, "SAME", [1, 256, 256, 3], [1, 16, 16, 1] )

# print patches
# print "Done."

# paths = [os.path.join(basedir,d) for d in ["A00_v2", "A01_v2", "A02_v2", "A03_v2", "A04_v2"]]
# files = [ (d,os.listdir(d)) for d in paths ]
# jpegs = [ os.path.join(d,f) for d,filesInDir in files for f in filesInDir if f.split('.')[-1] in ["jpg", "jpeg"]]

# #tensors = get_tensor_from_jpgs(jpegs)

# jpg = jpegs[0]
# sess = tf.InteractiveSession()

# queue = tf.train.string_input_producer([jpg])
# reader = tf.WholeFileReader()
# key,value = reader.read(queue)
# image = tf.expand_dims(tf.image.decode_jpeg(value),0)
# resized = tf.image.resize_bilinear(image, (16, 16))
# print resized.eval()
# print "Done."


# for t in tensors:
#     print t