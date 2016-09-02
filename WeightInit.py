import tensorflow as tf

class WeightInit:
    @staticmethod
    def truncatedNormal(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)    # Truncated normal means that values are dropped and re-drawn if they are more than 2 stds from mean
        return tf.Variable(initial)
    
    @staticmethod
    def zero(shape):
        return tf.Variable(tf.zeros(shape))

    @staticmethod
    def positive(shape, bias=0.):
        initial = tf.constant(bias, shape=shape)
        return tf.Variable(initial)