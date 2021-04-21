import tensorflow.compat.v1 as tf
import tensorflow_compression as tfc

import nll_compressor as nll_codec
import logisticmixturemodel as lmm
import bias_correction as bc

from utils.tools import read_png, quantize_image, write_png

from range_coder import RangeEncoder, RangeDecoder

import numpy as np
from PIL import Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

SCALE_MIN = 0.11
SCALE_MAX = 256
SCALES_LEVELS = 64

def compress(input_path, output_bin_path, output_res_path, ckp_dir, tau):

    with tf.device('/cpu:0'):
        # Load and Pad Image
        x = read_png(input_path)

        mod = tf.constant([64, 64, 1], dtype=tf.int32)
        div = tf.cast(tf.math.ceil(tf.math.truediv(tf.shape(x), mod)), tf.int32)
        paddings = tf.math.subtract(tf.math.multiply(div, mod), tf.shape(x))
        paddings = tf.expand_dims(paddings, 1)
        paddings = tf.concat([tf.convert_to_tensor(np.zeros((3,1)), dtype=tf.int32), paddings], axis=1)

        x_pad = tf.pad(x, paddings, "REFLECT")
        x_pad = tf.expand_dims(x_pad, 0)
        x_pad.set_shape([1, None, None, 3])

        x = tf.expand_dims(x,0)
        x.set_shape([1, None, None, 3])
        x_shape = tf.shape(x)
        x_norm = x_pad / 255

        # instantiate model
        encoder = nll_codec.Encoder(192)
        decoder = nll_codec.Decoder(192)
        hyper_encoder = nll_codec.HyperEncoder(192)
        hyper_decoder_sigma = nll_codec.HyperDecoder(192)
        hyper_decoder_mu = nll_codec.HyperDecoder(192)
        entropy_parameters_sigma = nll_codec.EntropyParameters(192)
        entropy_parameters_mu = nll_codec.EntropyParameters(192)
        entropy_bottleneck = tfc.EntropyBottleneck()
        res_compressor = nll_codec.ResidualCompressor(128, 5)
        masked_conv = nll_codec.MaskedConv2d("A", 64, (5, 5), padding="VALID")
        res_compressor_cond = bc.ResidualCompressor_cond(128, 5)

        # build model and encode/decode
        y = encoder(x_norm)
        y_shape = tf.shape(y)
        z = hyper_encoder(y)
        side_string = entropy_bottleneck.compress(z) # encode z (including quantization)
        z_hat_decode = entropy_bottleneck.decompress(side_string, tf.shape(z)[1:], channels=192) # decode z (including dequantization)
        psi_sigma = hyper_decoder_sigma(z_hat_decode)
        psi_mu = hyper_decoder_mu(z_hat_decode)
        sigma = entropy_parameters_sigma(psi_sigma)
        mu = entropy_parameters_mu(psi_mu)
        scale_table = np.exp(np.linspace(
            np.log(SCALE_MIN), np.log(SCALE_MAX), SCALES_LEVELS))
        conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, mean=mu)
        string = conditional_bottleneck.compress(y) # encode y (including quantization)
        y_hat_decode = conditional_bottleneck.decompress(string) # decode y (including dequantization)
        x_hat, res_prior = decoder(y_hat_decode)
        x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]
        x_hat = tf.clip_by_value(x_hat, 0, 1)
        x_hat = tf.math.floor(x_hat * 255 + 0.5)
        res_prior = res_prior[:, :x_shape[1], :x_shape[2], :]

        res = x - x_hat
        res_q = tf.where(res>=0, (2*tau+1) * tf.math.floor((res+tau)/(2*tau+1)),
                         (2*tau+1) * tf.math.ceil((res-tau)/(2*tau+1)))
        tau_list = tf.constant([int(tau - 1)], tf.int32)
        cond = tf.one_hot(tau_list, 5)

        num_pixels = tf.cast(tf.reduce_prod(x_shape[:-1]), dtype=tf.float32)

        res_q_patch = tf.placeholder(dtype=tf.float32, shape=(1,5,5,3))
        res_prior_channel_num = 64
        res_prior_patch = tf.placeholder(dtype=tf.float32, shape=(1,1,1,res_prior_channel_num))
        res_q_vector = tf.placeholder(dtype=tf.float32, shape=(1,1,1,3))

        bin_sz = 2 * tau + 1
        pmf_length = int(510 // bin_sz + 1)
        pmf_end = (255 // bin_sz) * bin_sz

        context = masked_conv(res_q_patch)
        res_prior_context = tf.concat([res_prior_patch, context], 3)

        bias_correction = True
        if bias_correction and int(tau) > 0:
            res_mu, res_log_sigma, res_pi, res_lambda = res_compressor_cond(res_prior_context, cond)
        else:
            res_mu, res_log_sigma, res_pi, res_lambda = res_compressor(res_prior_context)

        res_mu_tiled = tf.tile(res_mu, tf.constant([pmf_length, 1, 1, 1], tf.int32))
        res_log_sigma_tiled = tf.tile(res_log_sigma, tf.constant([pmf_length, 1, 1, 1], tf.int32))
        res_pi_tiled = tf.tile(res_pi, tf.constant([pmf_length, 1, 1, 1], tf.int32))
        res_lambda_tiled = tf.tile(res_lambda, tf.constant([pmf_length, 1, 1, 1], tf.int32))
        res_bottleneck = lmm.LogisticMixtureModel(res_mu_tiled, res_log_sigma_tiled, res_pi_tiled, res_lambda_tiled)
        res_pmf = res_bottleneck.pmf_tau(res_q_vector, tau)

        # MSE
        eval_mse = tf.reduce_mean(tf.squared_difference(x, x_hat))

        # PSNR
        eval_psnr = 10 * tf.math.log(255**2/eval_mse)/tf.math.log(10.0)

        # max abs diff
        eval_max_abs_diff = tf.reduce_max(tf.abs(tf.subtract(x, x_hat)))

        with tf.Session() as sess:

            latest = tf.train.latest_checkpoint(checkpoint_dir = ckp_dir)
            tf.train.Saver().restore(sess, save_path = latest)
            tensors = [string, side_string,
                       tf.shape(x)[1:-1], tf.shape(y)[1:-1], tf.shape(z)[1:-1]]
            arrays = sess.run(tensors)

            # write binary file
            packed = tfc.PackedTensors()
            packed.pack(tensors, arrays)
            with open(output_bin_path, "wb") as f:
                f.write(packed.string)

            # Lossy Image Encoding
            print("Lossy Image Encoding Start.")
            res_prior_out, res_q_out, _, x_org, x_out, lossy_mse, lossy_psnr, lossy_max_abs_diff, num_pixels_out, x_shape_out = sess.run(
                [res_prior, res_q, string, x, x_hat, eval_mse, eval_psnr, eval_max_abs_diff, num_pixels, x_shape])
            print("Lossy Image Encoding Finish.")

            k_sz = 5
            pad_sz = 2
            _, x_h, x_w, x_c = x_shape_out
            res_q_padded = np.pad(res_q_out, ((0,0), (pad_sz,pad_sz), (pad_sz,pad_sz), (0,0)), 'constant')

            encoder = RangeEncoder(output_res_path)
            print('Residual Encoding Start.')
            for h_idx in range(x_h):
                for w_idx in range(x_w):
                    res_q_extracted = res_q_padded[:, h_idx:h_idx+k_sz, w_idx:w_idx+k_sz, :]
                    res_prior_extracted = res_prior_out[:, h_idx, w_idx, :].reshape(1,1,1,res_prior_channel_num)
                    res_q_vector_extracted = res_q_out[:, h_idx, w_idx, :].reshape(1,1,1,3)

                    res_pmf_out = sess.run(res_pmf,
                                           feed_dict={res_q_patch:res_q_extracted, res_prior_patch:res_prior_extracted, res_q_vector:res_q_vector_extracted})
                    res_q_vector_extracted = (res_q_vector_extracted[0,0,0,:] + pmf_end)//bin_sz
                    for c_idx in range(x_c):
                        c_pmf = res_pmf_out[:, 0, 0, c_idx]
                        c_pmf = np.clip(c_pmf, 1.0/65025, 1.0)
                        c_pmf = c_pmf / np.sum(c_pmf)
                        cumFreq = np.floor(np.append([0.], np.cumsum(c_pmf)) * 65536. + 0.5).astype(np.int32).tolist()
                        encoder.encode([int(res_q_vector_extracted[c_idx])], cumFreq)
            print("Encoding Finish.")
            encoder.close()

        print("Lossy MSE:{}, Lossy PSNR:{}, Lossy max_abs_diff:{}".format(lossy_mse, lossy_psnr, lossy_max_abs_diff))
       
        img_sz_out = os.path.getsize(output_bin_path)
        res_sz_out = os.path.getsize(output_res_path)
        eval_sz_out = img_sz_out + res_sz_out
        img_bpsp = os.path.getsize(output_bin_path)*8/(x_c*x_h*x_w)
        res_bpsp = os.path.getsize(output_res_path)*8/(x_c*x_h*x_w)
        eval_bpsp = img_bpsp + res_bpsp
        
        print("tau:{}, bpsp:{}, img_bpsp:{}, res_bpsp:{}".format(tau, eval_bpsp, img_bpsp, res_bpsp))
        
        x_rec = np.clip(np.squeeze(x_out + res_q_out), 0, 255)
        max_abs_diff = np.amax(np.abs(x_org - x_rec))        
        mse = np.mean((x_org - x_rec)**2)
        psnr = 10*np.log10(255**2/mse)
        print("Max abs diff:{}, NLL MSE:{}, NLL PSNR:{}".format(max_abs_diff,mse,psnr))

    return eval_sz_out, img_sz_out, res_sz_out


def decompress(input_bin_path, input_res_path, output_img_path, ckp_dir, tau):

    with tf.device('/cpu:0'):
        # Load bin and res
        string = tf.placeholder(tf.string, [1])
        side_string = tf.placeholder(tf.string, [1])
        x_shape = tf.placeholder(tf.int32, [2])
        y_shape = tf.placeholder(tf.int32, [2])
        z_shape = tf.placeholder(tf.int32, [2])
        with open(input_bin_path, "rb") as f:
            packed = tfc.PackedTensors(f.read())
        tensors = [string, side_string, x_shape, y_shape, z_shape]
        arrays = packed.unpack(tensors)

        # instantiate model
        decoder = nll_codec.Decoder(192)
        hyper_decoder_sigma = nll_codec.HyperDecoder(192)
        hyper_decoder_mu = nll_codec.HyperDecoder(192)
        entropy_parameters_sigma = nll_codec.EntropyParameters(192)
        entropy_parameters_mu = nll_codec.EntropyParameters(192)
        entropy_bottleneck = tfc.EntropyBottleneck(dtype=tf.float32)
        res_compressor = nll_codec.ResidualCompressor(128, 5)
        masked_conv = nll_codec.MaskedConv2d("A", 64, (5, 5), padding="VALID")
        res_compressor_cond = bc.ResidualCompressor_cond(128, 5)

        # build decoder
        z_shape = tf.concat([z_shape, [192]], axis=0)
        z_hat_decode = entropy_bottleneck.decompress(side_string, z_shape, channels=192)  # decode z (including dequantization)
        psi_sigma = hyper_decoder_sigma(z_hat_decode)
        psi_mu = hyper_decoder_mu(z_hat_decode)
        sigma = entropy_parameters_sigma(psi_sigma)
        mu = entropy_parameters_mu(psi_mu)
        sigma = sigma[:, :y_shape[0], :y_shape[1], :]
        mu = mu[:, :y_shape[0], :y_shape[1], :]
        scale_table = np.exp(np.linspace(
            np.log(SCALE_MIN), np.log(SCALE_MAX), SCALES_LEVELS))
        conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, mean=mu, dtype=tf.float32)
        y_hat_decode = conditional_bottleneck.decompress(string)  # decode y (including dequantization)
        x_hat, res_prior = decoder(y_hat_decode)
        x_hat = x_hat[:, :x_shape[0], :x_shape[1], :]
        x_hat = tf.clip_by_value(x_hat, 0, 1)
        x_hat = tf.math.floor(x_hat * 255 + 0.5)
        res_prior = res_prior[:, :x_shape[0], :x_shape[1], :]

        tau_list = tf.constant([int(tau - 1)], tf.int32)
        cond = tf.one_hot(tau_list, 5)

        num_pixels = tf.cast(tf.reduce_prod(x_shape[:-1]), dtype=tf.float32)

        res_q_patch = tf.placeholder(dtype=tf.float32, shape=(1, 5, 5, 3))
        res_prior_channel_num = 64
        res_prior_patch = tf.placeholder(dtype=tf.float32, shape=(1, 1, 1, res_prior_channel_num))
        res_q_vector = tf.placeholder(dtype=tf.float32, shape=(1, 1, 1, 3))

        bin_sz = 2 * tau + 1
        pmf_length = int(510 // bin_sz + 1)
        pmf_end = (255 // bin_sz) * bin_sz

        context = masked_conv(res_q_patch)
        res_prior_context = tf.concat([res_prior_patch, context], 3)

        bias_correction = True
        if bias_correction and int(tau) > 0:
            res_mu, res_log_sigma, res_pi, res_lambda = res_compressor_cond(res_prior_context, cond)
        else:
            res_mu, res_log_sigma, res_pi, res_lambda = res_compressor(res_prior_context)

        res_mu_tiled = tf.tile(res_mu, tf.constant([pmf_length, 1, 1, 1], tf.int32))
        res_log_sigma_tiled = tf.tile(res_log_sigma, tf.constant([pmf_length, 1, 1, 1], tf.int32))
        res_pi_tiled = tf.tile(res_pi, tf.constant([pmf_length, 1, 1, 1], tf.int32))
        res_lambda_tiled = tf.tile(res_lambda, tf.constant([pmf_length, 1, 1, 1], tf.int32))
        res_bottleneck = lmm.LogisticMixtureModel(res_mu_tiled, res_log_sigma_tiled, res_pi_tiled, res_lambda_tiled)
        res_pmf = res_bottleneck.pmf_tau(res_q_vector, tau)

        with tf.Session() as sess:

            latest = tf.train.latest_checkpoint(checkpoint_dir=ckp_dir)
            tf.train.Saver().restore(sess, save_path=latest)

            # lossy image decoding  
            print("Lossy Image Decoding Start.")
            res_prior_out, x_out, num_pixels_out, x_shape_out = sess.run(
                [res_prior, x_hat, num_pixels, x_shape], feed_dict=dict(zip(tensors, arrays)))
            print("Lossy Image Decoding Finish.")

            k_sz = 5
            pad_sz = 2
            x_h, x_w = x_shape_out
            x_c = 3

            res_q_dec_padded = np.zeros((1, x_h + 2 * pad_sz, x_w + 2 * pad_sz, x_c))
            decoder = RangeDecoder(input_res_path)
            print('Residual Decoding Start.')
            for h_idx in range(x_h):
                for w_idx in range(x_w):
                    res_q_extracted = res_q_dec_padded[:, h_idx:h_idx + k_sz, w_idx:w_idx + k_sz, :]
                    res_prior_extracted = res_prior_out[:, h_idx, w_idx, :].reshape(1, 1, 1, res_prior_channel_num)

                    for c_idx in range(x_c):
                        res_q_vector_extracted = res_q_dec_padded[:, h_idx + pad_sz, w_idx + pad_sz, :].reshape(1, 1, 1, 3)
                        res_pmf_out = sess.run(res_pmf,
                                               feed_dict={res_q_patch: res_q_extracted,
                                                          res_prior_patch: res_prior_extracted,
                                                          res_q_vector: res_q_vector_extracted})
                        c_pmf = res_pmf_out[:, 0, 0, c_idx]
                        c_pmf = np.clip(c_pmf, 1.0 / 65025, 1.0)
                        c_pmf = c_pmf / np.sum(c_pmf)
                        cumFreq = np.floor(np.append([0.], np.cumsum(c_pmf)) * 65536. + 0.5).astype(
                            np.int32).tolist()
                        dataRec = decoder.decode(1, cumFreq)
                        res_q_dec_padded[0, h_idx + pad_sz, w_idx + pad_sz, c_idx] = dataRec[0] * bin_sz - pmf_end
            print("Decode Finish.")
            decoder.close()

        res_q_dec = res_q_dec_padded[:, pad_sz:x_h + pad_sz, pad_sz:x_w + pad_sz, :]

        x_rec = np.clip(np.squeeze(x_out + res_q_dec), 0, 255)
        im = Image.fromarray(np.uint8(x_rec))
        im.save(output_img_path)
        return x_rec



if __name__=="__main__":
    ckp_dir = "./ckp_003"
    input_path = "./test_patch/kodim05_p128.png"   
    output_dir = "./results_test_patch"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_path = input_path
    output_bin_path = os.path.join(output_dir, "lossy.bin")
    output_img_path = os.path.join(output_dir, "rec.png")
    output_res_path = os.path.join(output_dir, "residual.bin")
    
    tau=1.0
    do_decompression=True
    
    assert tau>=0.0 and tau<=5.0, "tau must be in {0, 1, ..., 5}"

    # compress 
    sz, img_sz, res_sz = compress(img_path, output_bin_path, output_res_path, ckp_dir, tau)
    tf.reset_default_graph()

    # decompress
    if do_decompression:
        im_rec = decompress(output_bin_path, output_res_path, output_img_path, ckp_dir, tau)
        tf.reset_default_graph()

    # statistics calculation
    im = np.array(Image.open(img_path))
    total_pixels = np.prod(im.shape)
    
    bpsp = sz*8/total_pixels
    img_bpsp = img_sz*8/total_pixels
    res_bpsp = res_sz*8/total_pixels 
    print("\nimg_shape:{}".format(im.shape))
    print("tau:{}, bpsp:{}, img_bpsp:{}, res_bpsp:{}".format(tau, bpsp, img_bpsp, res_bpsp))

    if do_decompression:
        max_abs_diff = np.amax(np.abs(im.astype(np.float32) - im_rec.astype(np.float32)))        
        mse = np.mean((im.astype(np.float32) - im_rec.astype(np.float32))**2)
        psnr = 10*np.log10(255**2/mse)
        print("Decoded: Max abs diff:{}, NLL MSE:{}, NLL PSNR:{}".format(max_abs_diff,mse,psnr))
    

