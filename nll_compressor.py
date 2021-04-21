import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

import numpy as np

# from utils.tools import _debug_func

class ResBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.conv1 = tfc.SignalConv2D(self.num_filters//2, (1, 1), name="conv1", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu)
        self.conv2 = tfc.SignalConv2D(self.num_filters//2, (3, 3), name="conv2", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu)
        self.conv3 = tfc.SignalConv2D(self.num_filters, (1, 1), name="conv3", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu)
        super().build(input_shape)

    def call(self, input):
        output = input
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output) + input
        return output


class AnalysisBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.conv1 = tfc.SignalConv2D(self.num_filters, (3, 3), name="conv1", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu)
        self.conv2 = tfc.SignalConv2D(self.num_filters, (3, 3), name="conv2", corr=True, strides_down=2,
                             padding="same_zeros", use_bias=True,
                             activation=tfc.GDN(name="gdn"))
        self.conv_sc = tfc.SignalConv2D(self.num_filters, (3, 3), name="conv_sc", corr=True, strides_down=2,
                             padding="same_zeros", use_bias=True,
                             activation=None)
        self.conv3 = tfc.SignalConv2D(self.num_filters, (3, 3), name="conv3", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu)
        self.conv4 = tfc.SignalConv2D(self.num_filters, (3, 3), name="conv4", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu)
        super().build(input_shape)

    def call(self, input):
        output = input
        output = self.conv1(output)
        output = self.conv2(output)
        output2 = self.conv_sc(input) + output

        output = self.conv3(output2)
        output = self.conv4(output)
        output = output + output2
        return output


class SynthesisBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.conv4 = tfc.SignalConv2D(self.num_filters, (3, 3), name="conv4", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu)
        self.conv3 = tfc.SignalConv2D(self.num_filters, (3, 3), name="conv3", corr=False, strides_up=2,
                             padding="same_zeros", use_bias=True,
                             activation=tfc.GDN(name="igdn", inverse=True))
        self.conv_sc = tfc.SignalConv2D(self.num_filters, (3, 3), name="conv_sc", corr=False, strides_up=2,
                             padding="same_zeros", use_bias=True,
                             activation=None)
        self.conv2 = tfc.SignalConv2D(self.num_filters, (3, 3), name="conv2", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu)
        self.conv1 = tfc.SignalConv2D(self.num_filters, (3, 3), name="conv1", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu)
        super().build(input_shape)

    def call(self, input):
        output = input
        output = self.conv1(output)
        output = self.conv2(output)
        output2 = input + output

        output = self.conv3(output2)
        output = self.conv4(output)
        output = output + self.conv_sc(output2)
        return output


class AttentionBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.layers1 = [
            ResBlock(self.num_filters, name="res1_1"),
            ResBlock(self.num_filters, name="res1_2"),
            ResBlock(self.num_filters, name="res1_3")
        ]
        self.layers2 = [
            ResBlock(self.num_filters, name="res2_1"),
            ResBlock(self.num_filters, name="res2_2"),
            ResBlock(self.num_filters, name="res2_3")
        ]
        self.conv = tfc.SignalConv2D(self.num_filters, (1, 1), name="conv", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=None)
        super().build(input_shape)

    def call(self, input):
        output = input
        for layer in self.layers1:
            output = layer(output)

        output2 = input
        for layer in self.layers2:
            output2 = layer(output2)
        output2 = tf.math.sigmoid(self.conv(output2))
        output = output * output2 + input
        return output


class Encoder(tf.keras.layers.Layer):

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.layers = [
            AnalysisBlock(self.num_filters, name="analysis1"),
            AnalysisBlock(self.num_filters, name="analysis2"),
            AttentionBlock(self.num_filters, name="attention1"),
            AnalysisBlock(self.num_filters, name="analysis3"),
            tfc.SignalConv2D(self.num_filters, (3, 3), name="conv", corr=True, strides_down=2,
                             padding="same_zeros", use_bias=True,
                             activation=None),
            AttentionBlock(self.num_filters, name="attention2"),
        ]
        super().build(input_shape)

    def call(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.layers = [
            AttentionBlock(self.num_filters, name="attention1"),
            SynthesisBlock(self.num_filters, name="synthesis1"),
            SynthesisBlock(self.num_filters, name="synthesis2"),
            AttentionBlock(self.num_filters, name="attention2"),
            SynthesisBlock(self.num_filters, name="synthesis3"),
        ]
        self.conv_img = tfc.SignalConv2D(3, (3, 3), name="conv_img", corr=False, strides_up=2,
                             padding="same_zeros", use_bias=True,
                             activation=None)
        self.conv_prior = tfc.SignalConv2D(64, (3, 3), name="conv_prior", corr=False, strides_up=2,
                             padding="same_zeros", use_bias=True,
                             activation=None)
        super().build(input_shape)

    def call(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)

        rec_img = self.conv_img(output)
        prior = self.conv_prior(output)
        return rec_img, prior


class HyperEncoder(tf.keras.layers.Layer):

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.layers = [
            tfc.SignalConv2D(self.num_filters, (3, 3), name="layer_0", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (3, 3), name="layer_1", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (3, 3), name="layer_2", corr=True, strides_down=2,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (3, 3), name="layer_3", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (3, 3), name="layer_4", corr=True, strides_down=2,
                             padding="same_zeros", use_bias=True,
                             activation=None),
        ]
        super().build(input_shape)

    def call(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


class HyperDecoder(tf.keras.layers.Layer):

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.layers = [
            tfc.SignalConv2D(self.num_filters, (3, 3), name="layer_0", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (3, 3), name="layer_1", corr=False, strides_up=2,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (3, 3), name="layer_2", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (3, 3), name="layer_3", corr=False, strides_up=2,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (3, 3), name="layer_4", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=None),
        ]
        super().build(input_shape)

    def call(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output


class EntropyParameters(tf.keras.layers.Layer):

    def __init__(self, num_filters, *args, **kwargs):
        self.num_filters = num_filters
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.layers = [
            tfc.SignalConv2D(self.num_filters, (1, 1), name="layer_0", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (1, 1), name="layer_1", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (1, 1), name="layer_2", corr=True, strides_down=1,
                             padding="same_zeros", use_bias=True,
                             activation=None),
        ]
        super().build(input_shape)

    def call(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

class TailBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, num_mixtures, *args, **kwargs):
        self.num_filters = num_filters
        self.num_mixtures = num_mixtures
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.layers1 = [
            tfc.SignalConv2D(self.num_filters, (1, 1), name="layer_0", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_filters, (1, 1), name="layer_1", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu)
        ]
        self.layers2 = [
            tfc.SignalConv2D(self.num_filters, (1, 1), name="layer_2", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=tf.nn.leaky_relu),
            tfc.SignalConv2D(self.num_mixtures * 3, (1, 1), name="layer_3", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=None)
        ]
        self.conv_sc = tfc.SignalConv2D(self.num_filters, (1, 1), name="conv_sc", corr=False, strides_up=1,
                             padding="same_zeros", use_bias=True,
                             activation=None)
        super().build(input_shape)

    def call(self, input):
        output = input
        for layer in self.layers1:
            output = layer(output)

        output += self.conv_sc(input)

        for layer in self.layers2:
            output = layer(output)
        return output


class MaskedConv2d(tf.keras.layers.Layer):

    def __init__(self,
                 mask_type,
                 num_filters,
                 kernel_size,
                 strides=(1, 1),
                 padding="SAME",
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 *args,
                 **kwargs):
        assert mask_type == 'A' or mask_type == 'B'
        self.mask_type = mask_type
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.use_bias = use_bias
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.num_filters),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.num_filters,), initializer=self.bias_initializer, trainable=True)

        mask = np.zeros(shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1], self.num_filters), dtype=np.float32)

        kh = self.kernel_size[0]
        kw = self.kernel_size[1]
        mask[:kh//2, :, :, :] = 1
        mask[kh//2, :kw//2, :, :] = 1
        if self.mask_type == "B":
            mask[kh//2, kw//2, :, :] = 1
        self.mask = tf.constant(mask, dtype=tf.float32)

        super().build(input_shape)

    def call(self, input):
        out = tf.nn.conv2d(input, self.weight * self.mask, self.strides, self.padding)
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)
        return out


class ResidualCompressor(tf.keras.layers.Layer):

    def __init__(self, num_filters, num_mixtures, *args, **kwargs):
        self.num_filters = num_filters
        self.num_mixtures = num_mixtures
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.tail_mu = TailBlock(self.num_filters, self.num_mixtures, name="tail_mu")
        self.tail_log_sigma = TailBlock(self.num_filters, self.num_mixtures, name="tail_log_sigma")
        self.tail_pi = TailBlock(self.num_filters, self.num_mixtures, name="tail_pi")
        self.tail_lambda = TailBlock(self.num_filters, self.num_mixtures, name="tail_lambda")
        super().build(input_shape)

    def call(self, input):
        mu = self.tail_mu(input)
        log_sigma = self.tail_log_sigma(input)
        pi = self.tail_pi(input)
        lamda = self.tail_lambda(input)

        return mu, log_sigma, pi, lamda


