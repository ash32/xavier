import tensorflow as tf
import numpy as np
from xavier.models.cnn_eiie import ConvEIIE


class CnnTest(tf.test.TestCase):
    @staticmethod
    def np_assert(a, b):
        np.testing.assert_allclose(a, b, rtol=1e-6)

    def test_output_layer(self):
        cnn = ConvEIIE()
        x_val = np.random.rand(1, 5, 3, 1)

        X = tf.placeholder(tf.float32, shape=x_val.shape)
        output = cnn._output_layer(X)

        with self.test_session() as sess:
            tf.global_variables_initializer().run()
            out_val = sess.run(output, feed_dict={X: x_val})

        x = np.concatenate([np.zeros((5, 1)), x_val.squeeze()], axis=1)
        x = np.exp(x - x.max(axis=1).reshape((-1, 1)))
        x = x / np.sum(x, axis=1).reshape((-1, 1))

        self.np_assert(x, out_val)

    def test_rewards(self):
        cnn = ConvEIIE()

        x_val = np.random.rand(5, 4)
        y_val = np.random.rand(5, 4)

        X = tf.placeholder(tf.float32, shape=x_val.shape)
        y = tf.placeholder(tf.float32, shape=y_val.shape)
        output = cnn._reward_layer(X, y)

        with self.test_session() as sess:
            out_val = sess.run(output, feed_dict={X: x_val, y: y_val})

        x = np.mean(np.log(np.sum(x_val * y_val, axis=1)))

        self.np_assert(x, out_val)

