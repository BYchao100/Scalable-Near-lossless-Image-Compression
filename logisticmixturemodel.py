import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

import numpy as np

class LogisticMixtureModel(tf.keras.layers.Layer):

    def __init__(self, mean, log_sigma, mixture_weights, autoregression_coefficients=None, *args, **kwargs):
        #
        super().__init__(*args, **kwargs)
        self.mean = tf.convert_to_tensor(mean)
        mean_shape = tf.shape(self.mean)
        self.mix_num = mean_shape[-1]//3
        self.mean = tf.reshape(self.mean, tf.concat([mean_shape[:-1],[3],[self.mix_num]],0))
        #
        self.log_sigma = tf.convert_to_tensor(log_sigma)
        log_sigma_shape = tf.shape(self.log_sigma)
        self.log_sigma = tf.reshape(self.log_sigma, tf.concat([log_sigma_shape[:-1],[3],[self.mix_num]],0))
        self.log_sigma = tf.maximum(self.log_sigma, -7.)
        #
        self.mixture_weights = tf.convert_to_tensor(mixture_weights)
        mw_shape = tf.shape(self.mixture_weights)
        self.mixture_weights = tf.reshape(self.mixture_weights, tf.concat([mw_shape[:-1],[3],[self.mix_num]],0))
        #
        if autoregression_coefficients is not None:
            self.coeffs = tf.math.tanh(autoregression_coefficients)
            coeffs_shape = tf.shape(self.coeffs)
            self.coeffs = tf.reshape(self.coeffs, tf.concat([coeffs_shape[:-1],[3],[self.mix_num]],0))
        else:
            self.coeffs = None

    def build(self, input_shape):
        #
        super().build(input_shape)

    def pmf_tau(self, input, tau):
        input_shape = tf.shape(input)
        half = tf.constant(.5+tau, dtype=self.dtype)

        bin_sz = 2 * tau + 1
        samples_num = int(510 // bin_sz + 1)
        sample_end = (255 // bin_sz) * bin_sz
        samples = np.arange(-sample_end, sample_end + 1, step=bin_sz).reshape(samples_num, 1, 1, 1) + np.zeros((samples_num, 1, 1, 3))
        samples = tf.constant(samples.reshape(samples_num, 1, 1, 3, 1) + np.zeros((samples_num, 1, 1, 3, 5)), tf.float32)

        x = tf.reshape(input, tf.concat([input_shape, [1]],0)) + tf.zeros(tf.concat([input_shape, [self.mix_num]],0))
        x = tf.tile(x, tf.constant([samples_num, 1, 1, 1, 1], tf.int32))

        if self.coeffs is not None:
            m1 = tf.reshape(self.mean[:, :, :, 0, :], tf.concat([[samples_num, 1, 1],[1],[self.mix_num]],0))
            m2 = self.mean[:, :, :, 1, :] + self.coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]
            m2 = tf.reshape(m2, tf.concat([[samples_num, 1, 1],[1],[self.mix_num]],0))
            m3 = self.mean[:, :, :, 2, :] + self.coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + \
                                       self.coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]
            m3 = tf.reshape(m3, tf.concat([[samples_num, 1, 1],[1],[self.mix_num]],0))
            self.mean = tf.concat([m1, m2, m3], 3)

        centered_samples = samples - self.mean
        inv_sigma = tf.exp(-self.log_sigma)
        plus_in = inv_sigma * (centered_samples + half)
        cdf_plus = tf.math.sigmoid(plus_in)
        min_in = inv_sigma * (centered_samples - half)
        cdf_min = tf.math.sigmoid(min_in)
        log_one_minus_cdf_min = -tf.math.softplus(min_in) # 255
        log_cdf_plus = plus_in - tf.math.softplus(plus_in) # -255
        cdf_delta = cdf_plus - cdf_min

        log_probs = tf.where(samples - half < -254.999, log_cdf_plus,
                             tf.where(samples + half > 254.999, log_one_minus_cdf_min,
                                      tf.log(tf.maximum(cdf_delta, 1e-5))))

        log_probs = log_probs + self.log_prob_from_logits(self.mixture_weights)

        return tf.exp(self.log_sum_exp(log_probs))

    def log_prob_from_logits(self, x):
        axis = len(x.get_shape())-1
        m = tf.reduce_max(x, axis, keepdims=True)
        return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))

    def log_sum_exp(self, x):
        axis = len(x.get_shape())-1
        m = tf.reduce_max(x, axis)
        m2 = tf.reduce_max(x, axis, keepdims=True)
        return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))



