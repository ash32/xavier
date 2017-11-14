from xavier.models.base import TrainableModel

import tensorflow as tf


class ConvEIIE(TrainableModel):
    X = y = None
    output = None
    training_op = None
    var_init = None
    saver = None

    @staticmethod
    def conv_layers(X, num_periods):
        kernel_height1 = 3
        conv1 = tf.layers.conv2d(X, filters=2, kernel_size=(kernel_height1, 1), name='conv1')
        relu1 = tf.nn.relu(conv1, name='relu1')

        kernel_height2 = num_periods - kernel_height1 + 1
        conv2 = tf.layers.conv2d(relu1, filters=20, kernel_size=(kernel_height2, 1), name='conv2')
        relu2 = tf.nn.relu(conv2, name='relu2')

        return tf.layers.conv2d(relu2, filters=1, kernel_size=(1, 1), name='conv3')

    @staticmethod
    def output_layer(X):
        cash_bias = tf.Variable([0.], name='cash_bias')

        squeezed = tf.squeeze(X)
        bias_broadcast = tf.expand_dims(tf.tile(cash_bias, tf.shape(squeezed)[:1]), 1)

        stacked = tf.concat([bias_broadcast, squeezed], axis=1, name='logits')
        return tf.nn.softmax(stacked, name='portfolio_weights')

    @staticmethod
    def reward_layer(X, y):
        trial_rewards = tf.log(tf.reduce_sum(tf.multiply(X, y), axis=1), name='trial_rewards')
        return tf.reduce_mean(trial_rewards, name='total_reward')

    def build_model(self, num_noncash_assets, num_periods):
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(1, None, num_noncash_assets, 3), name='X')
            self.y = tf.placeholder(tf.float32, shape=(None, num_noncash_assets + 1), name='y')
            # w = tf.placeholder(tf.float32, shape=(1, num_assets, None, 3), name='w')

        with tf.name_scope('cnn'):
            conv_layers = self.conv_layers(self.X, num_periods)

        with tf.name_scope('output'):
            self.output = self.output_layer(conv_layers)

        with tf.name_scope('reward'):
            total_reward = self.reward_layer(self.output, self.y)

        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer()
            self.training_op = optimizer.minimize(-total_reward, name='training_op')

        with tf.name_scope('init_and_save'):
            self.var_init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def get_distribution_weights(self, closing_price_vec, prev_weight_vec):
        with tf.Session() as sess:
            self.var_init.run()
            out = sess.run([self.output], feed_dict={self.X: closing_price_vec})

        return out
