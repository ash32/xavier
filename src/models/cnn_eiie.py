from .base import BaseModel

import tensorflow as tf

class ConvEIIE(BaseModel):
    def __init__(self, num_assets, num_periods):
        self.build_model(num_assets, num_periods)

    def build_model(self, num_assets, num_periods):
        with tf.name_scope('inputs'):
            X = tf.placeholder(tf.float32, shape=(1, num_assets, None, 3))
            # w = tf.placeholder(tf.float32, shape=(1, num_assets, None, 3))

        with tf.name_scope('conv2d'):
            kernel_width1 = 3
            conv1 = tf.layers.conv2d(X, filters=2, kernel_size=(1, kernel_width1), name='conv1')

            kernel_width2 = num_periods - kernel_width1 + 1
            conv2 = tf.layers.conv2d(conv1, filters=20, kernel_size=(1, kernel_width2), name='conv2')

            conv3 = tf.layers.conv2d(conv2, filters=1, kernel_size=(1, 1), name='conv3')

        with tf.name_scope('output'):
            cash_bias = tf.Variable(0, name='cash_bias')

            squeezed = tf.squeeze(conv3)
            bias_broadcast = tf.tile(cash_bias, tf.shape(squeezed)[1:])


