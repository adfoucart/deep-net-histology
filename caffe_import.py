import tensorflow as tf

def caffe_import(in_, out_):
    sess = tf.Session()
    with open(in_, "rb") as f:
        gd = tf.GraphDef()
        gd.ParseFromString(f.read())

    ng = tf.Graph()

    with ng.as_default():
        for node in gd.node:
            if node.name.find("softmax") < 0:
                tf.ops.convert_to_tensor(node, name=node.name)

    