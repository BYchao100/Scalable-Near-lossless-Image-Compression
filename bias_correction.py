import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

import numpy as np

import nll_compressor as nll_codec


class Conv2d_cond(tf.keras.layers.Layer):

    def __init__(self, num_filters, kernel_size, *args, **kwargs):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.weight_cond = tf.keras.layers.Dense(self.num_filters, activation=tf.math.softplus, use_bias=False)
        self.bias_cond = tf.keras.layers.Dense(self.num_filters, use_bias=False)
        self.conv = tfc.SignalConv2D(self.num_filters, self.kernel_size, name="conv", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=False,
                             activation=None)
        super().build(input_shape)

    def call(self, input, cond):
        cond = tf.cast(cond, tf.float32)
        output = self.conv(input)
        w_cond = tf.expand_dims(tf.expand_dims(self.weight_cond(cond), axis=1), axis=2)
        b_cond = tf.expand_dims(tf.expand_dims(self.bias_cond(cond), axis=1), axis=2)
        output = tf.math.multiply(output, w_cond)
        output = tf.math.add(output, b_cond)

        return output


class TailBlock_cond(tf.keras.layers.Layer):

    def __init__(self, num_filters, num_mixtures, *args, **kwargs):
        self.num_filters = num_filters
        self.num_mixtures = num_mixtures
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.conv_cond1 = Conv2d_cond(self.num_filters, (1, 1))
        self.conv_cond2 = Conv2d_cond(self.num_filters, (1, 1))
        self.conv_cond3 = Conv2d_cond(self.num_filters, (1, 1))
        self.conv_cond4 = Conv2d_cond(self.num_mixtures * 3, (1, 1))
        self.conv_sc_cond = Conv2d_cond(self.num_filters, (1, 1))

        super().build(input_shape)

    def call(self, input, cond):
        output = tf.nn.leaky_relu(self.conv_cond1(input, cond))
        output = tf.nn.leaky_relu(self.conv_cond2(output, cond))

        output += self.conv_sc_cond(input, cond)

        output = tf.nn.leaky_relu(self.conv_cond3(output, cond))
        output = self.conv_cond4(output, cond)

        return output


class ResidualCompressor_cond(tf.keras.layers.Layer):

    def __init__(self, num_filters, num_mixtures, *args, **kwargs):
        self.num_filters = num_filters
        self.num_mixtures = num_mixtures
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.tail_mu = TailBlock_cond(self.num_filters, self.num_mixtures, name="tail_mu")
        self.tail_log_sigma = TailBlock_cond(self.num_filters, self.num_mixtures, name="tail_log_sigma")
        self.tail_pi = TailBlock_cond(self.num_filters, self.num_mixtures, name="tail_pi")
        self.tail_lambda = TailBlock_cond(self.num_filters, self.num_mixtures, name="tail_lambda")
        super().build(input_shape)

    def call(self, input, cond):
        mu = self.tail_mu(input, cond)
        log_sigma = self.tail_log_sigma(input, cond)
        pi = self.tail_pi(input, cond)
        lamda = self.tail_lambda(input, cond)

        return mu, log_sigma, pi, lamda





