import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
import tensorflow.keras.layers as L

from .datasets import *

_tf_int_dtype = tf.int32
_np_int_dtype = np.int32

def circular_bitshift(
    x, 
    shift, 
    max_bits,
    numpy_out=False,
    dtype=_tf_int_dtype,
):
    shift = tf.math.mod(shift, max_bits)
    val = tf.bitwise.bitwise_or(
        tf.bitwise.right_shift(x, shift),
        tf.math.mod(
            tf.bitwise.left_shift(x, max_bits-shift),
            tf.math.pow(2, max_bits)
        )
    )
    val = tf.cast(val, dtype)
    if numpy_out:
        val = val.numpy()
    return val

class CircularBitshiftEmbedding(L.Layer):
    
    def __init__(
        self,
        shifts,
        total_bit_width,
        sub_bit_width,
        sub_dimension,
    ):
        super().__init__()
        self.total_bit_width = total_bit_width
        self.sub_bit_width = sub_bit_width
        self.shifts = shifts
        nvocab = 2**sub_bit_width
        self.nvocab = nvocab
        
        embeddings = []
        for shift in shifts:
            embeddings.append(
                L.Embedding(
                    nvocab,
                    sub_dimension,
                )
            )
        self.embedding_ls = embeddings
        self.concat_l = L.Concatenate()
        
    
    def call(self, x):
        vecs = []
        for idx, shift in enumerate(self.shifts):
            indexes = tf.math.mod(
                circular_bitshift(
                    x,
                    shift,
                    max_bits=self.total_bit_width,
                ),
                self.nvocab,
            )
            vecs.append(self.embedding_ls[idx](indexes))
        return self.concat_l(vecs)


def cos_squared_schedule(
    t, 
    Tmax=1.0,
    ns=2e-4, 
    ds=2.5e-4,
):
    rt = t/Tmax
    return np.cos(((rt+ns)/(1+ds))*np.pi/2)**2

def int2bits(x, n_bits, dtype=_tf_int_dtype):
    x = tf.bitwise.right_shift(tf.expand_dims(x, -1), tf.cast(tf.range(n_bits), dtype))
    x = tf.math.mod(x, 2)
    return x

def bits2int(x, dtype=_tf_int_dtype):
    x = tf.cast(x, dtype)
    n = x.shape[-1]
    x = tf.math.reduce_sum(x*(2**tf.cast(tf.range(n), dtype=dtype)), -1)
    return x


class AnalogBitsTransform(object):
    
    def __init__(
        self,
        n_bits,
        scale_factor=1.0,
        numpy_out=True,
    ):
        self.n_bits = n_bits
        self.scale_factor=scale_factor
        self.numpy_out = numpy_out
        
    def __call__(self, x):
        bits = int2bits(x, n_bits=self.n_bits)
        if self.numpy_out:
            bits = bits.numpy()
        return ((2.0*self.scale_factor) * bits) - self.scale_factor
    
    def inverse_transform(self, x):
        return bits2int(x > 0.0)

class GaussianDiffusion(object):
    
    def __init__(
        self,
        schedule_fn,
        add_positions=True,
    ):
        self.schedule_fn = schedule_fn
        self.add_positions = add_positions

    def __call__(self, x):
        tvals = np.random.uniform(0, 1, size=x.shape[0])
        gammas = self.schedule_fn(tvals)[:, np.newaxis, np.newaxis]
        
        noise_direction = np.random.normal(size=x.shape)
        x_crpt = np.sqrt(gammas)*x + np.sqrt(1-gammas)*noise_direction
        
        delta = x_crpt - x

        xvals = [x_crpt, tvals]
        yvals = [x, delta]
        
        if self.add_positions:
            positions = np.repeat(np.arange(x.shape[1])[np.newaxis], x.shape[0], axis=0)
            xvals.append(positions)

        return xvals, yvals


def gen_random_bitmask(
    x,
    hot_fraction,
    n_bits,
    #dtype=np.uint32,
):
    as_bits = np.random.random(size=list(x.shape)+[n_bits]) < hot_fraction
    as_pwrs = as_bits*2**np.arange(n_bits)
    as_ints = np.sum(as_pwrs, axis=-1).astype(x.dtype)
    return as_ints

class BinaryDiffusion(object):
    
    def __init__(
        self,
        n_bits,
        schedule_fn,
        add_positions=True,
        y_as_bits=True,
    ):
        if n_bits >= 32:
            raise NotImplementedError("")
        
        self.n_bits = n_bits
        self.schedule_fn = schedule_fn
        self.add_positions = add_positions
        self.y_as_bits = y_as_bits
        
    def __call__(
        self,
        x,
    ):
        tvals = np.random.uniform(0, 1, size=x.shape[0])
        
        #gamma represents the variance share of the uncorrupted data
        gammas = self.schedule_fn(tvals)
        
        #allow this fraction of noise through
        corrupt_fracs = np.sqrt(1.0-gammas)[:, np.newaxis, np.newaxis]
        
        corruptable_mask = gen_random_bitmask(
            x,
            hot_fraction=corrupt_fracs,
            n_bits=self.n_bits
        ).astype(x.dtype)
        
        #generate some random bitnoise
        unfiltered_noise = np.random.randint(2**self.n_bits, size=x.shape, dtype=x.dtype)
        filtered_noise = np.bitwise_and(corruptable_mask, unfiltered_noise)
        
        noisy_x = np.bitwise_xor(x, filtered_noise)
        
        xvals = [noisy_x, tvals]
        yvals = [x, filtered_noise]
        
        if self.add_positions:
            positions = np.repeat(np.arange(x.shape[1])[np.newaxis], x.shape[0], axis=0)
            xvals.append(positions)
        
        if self.y_as_bits:
            yvals = [int2bits(yv, self.n_bits) for yv in yvals]
        
        return xvals, yvals
        

def gen_random_codewords(
    n_codewords, 
    n_bits,
    dtype=_np_int_dtype,
):
    max_codes = 2**n_bits
    assert max_codes >= n_codewords
    if n_codewords > max_codes/8:
        codes = np.random.permutation(max_codes)[:n_codewords]
    else:
        codes = np.unique(np.random.randint(max_codes, size=n_codewords))
        while len(codes) < n_codewords:
            codes = np.unique(np.hstack([
                codes,
                np.random.randint(
                    max_codes, 
                    size=n_codewords-len(codes)
                ),
            ]))
    
    return codes.astype(dtype)

def make_corrector_lookup(
    codewords, 
    n_bits,
    dtype=_np_int_dtype,
):
    code_bits = int2bits(codewords, n_bits=n_bits).numpy()
    
    all_msgs = np.arange(2**n_bits).astype(dtype)
    all_msgs_bits = int2bits(all_msgs, n_bits=n_bits).numpy()

    nnsearcher = NearestNeighbors().fit(code_bits)
    self_dists, _ = nnsearcher.kneighbors(code_bits, n_neighbors=2)
    
    dists, neighbor_indexes = nnsearcher.kneighbors(all_msgs_bits, n_neighbors=2)
    corrector_lookup = codewords[neighbor_indexes[:, 0]]
    
    info = dict(
        dists=dists,
        neighbor_indexes=neighbor_indexes,
        nnsearcher=nnsearcher,
        self_dists=self_dists,
    )
    
    return corrector_lookup, info


class LookupBinaryCoder(object):
    
    def __init__(
        self,
        codewords,
        code2code=None, #error corrector
    ):
        code_bits = int(np.ceil(np.log2(np.max(code2code))))
        self.vocab_size = len(codewords)
        self.code_bits = code_bits
        
        #the code words in order
        self.codewords = codewords
        
        #noiseless code to message 
        self.noiseless_code2msg = {code:msg for msg, code in enumerate(codewords)}
 
        #correct a corrupted code by indexing
        self.code2code = code2code

        self.noisy_code2msg = np.zeros(len(code2code), dtype=np.int32)
        for noisy_code in range(len(code2code)):
            corrected_code = code2code[noisy_code]
            msg = self.noiseless_code2msg[corrected_code]
            self.noisy_code2msg[noisy_code] = msg            
    
    def __call__(self, msg):
        return self.encode(msg)
    
    def inverse_transform(self, code):
        return self.decode(code)
    
    def encode(self, msg):
        return self.codewords[msg]
    
    def correct(self, code):
        return self.code2code[code]
    
    def decode(self, code):
        return self.noisy_code2msg[code]


class BlockCoder(object):
    
    def __init__(
        self,
        block_coders,
        vocab_size=None,
    ):
        self.block_coders = block_coders
        self.n_blocks = len(block_coders)
        self.index_shape = [coder.vocab_size for coder in block_coders]
        
        if vocab_size is None:
            vocab_size = np.prod(self.index_shape)
        self.vocab_size = vocab_size
        assert np.prod(self.index_shape) >= vocab_size
    
    def encode(self, msg):
        block_msgs = np.unravel_index(msg, shape=self.index_shape)
        encoded = np.zeros_like(msg)
        block_shift = 0
        for block_idx in range(self.n_blocks):
            coder = self.block_coders[block_idx]
            encoded += coder.encode(block_msgs[block_idx]) << block_shift
            block_shift += coder.code_bits
        return encoded
    
    def correct(self, code):
        raise NotImplementedError("not implemented :(")
        
    
    def decode(self, code):
        block_msgs = []
        block_shift = 0
        for block_idx in range(self.n_blocks):
            coder = self.block_coders[block_idx]
            block_msgs.append(
                coder.decode(
                    (code >> block_shift) % 2**coder.code_bits
                )
            )
            block_shift += coder.code_bits
        
        return np.ravel_multi_index(block_msgs, dims=self.index_shape)


def ddim_update(
    x_t,
    x_pred,
    gamma_now,
    gamma_next,
):  
    eps = (1.0/np.sqrt(1-gamma_now))*(x_t - np.sqrt(gamma_now)*x_pred)
    x_next = np.sqrt(gamma_next)*x_pred + np.sqrt(1.0-gamma_next)*eps
    return x_next

def ddpm_update(
    x_t,
    x_pred,
    gamma_now,
    gamma_next,
):
    alpha_now = gamma_now/gamma_next#gamma(t_now)/gamma(t_next)
    sigma_now = np.sqrt(1-alpha_now)
    z  = np.random.normal(size=x_t.shape)
    
    eps = (1.0/np.sqrt(1-gamma_now))*(x_t - np.sqrt(gamma_now)*x_pred)
    x_next = (1.0/np.sqrt(alpha_now))*(x_t - (1-alpha_now)/(np.sqrt(1-gamma_now))*eps) + sigma_now*z
    return x_next


def linear_gated_xor_update(
    x_t,
    x_pred,
    gamma_now,
    gamma_next,
    n_bits,
):
    bmask = gen_random_bitmask(
        x_t, 
        hot_fraction=gamma_now,
        n_bits=n_bits,
    )

    full_noise = np.bitwise_xor(x_t, x_pred)
    masked_noise = np.bitwise_and(full_noise, bmask)

    x_next = np.bitwise_xor(
        x_t,
        masked_noise,
    )

    return x_next

def sqrt_gated_xor_update(
    x_t,
    x_pred,
    gamma_now,
    gamma_next,
    n_bits,
):
    bmask = gen_random_bitmask(
        x_t, 
        hot_fraction=np.sqrt(gamma_now),
        n_bits=n_bits,
    )

    full_noise = np.bitwise_xor(x_t, x_pred)
    masked_noise = np.bitwise_and(full_noise, bmask)

    x_next = np.bitwise_xor(
        x_t,
        masked_noise,
    )

    return x_next

def sigmoid(logit):
    return 1.0/(1.0 + np.exp(-logit))

def generate_samples(
    seed_noise,
    model,
    schedule_fn,
    update_fn,
    n_steps=100,
    condition=None,
    condition_mask_fn=np.isnan,
    rel_td=0.02,
    input_signature="x",
    output_signature="y,n",
    model_output_type="continuous",
    estimate="x-n",
    return_trajectory=False,
):          
    x_t = seed_noise
    if not condition is None:
        x_t = np.where(condition_mask_fn(x_t), x_t, condition)
    
    if return_trajectory:
        trajectory = [x_t]
    
    td = rel_td*n_steps
    for step_idx in range(n_steps):
        t_now = 1-step_idx/n_steps
        t_next = max(1-(step_idx+1+td)/n_steps, 0)
        
        gamma_now = schedule_fn(t_now)
        gamma_next = schedule_fn(t_next)
        
        x_inputs = []
        if input_signature == "x":
            x_inputs = x_t
        elif input_signature == "x,p":
            pos = np.ones(x_t.shape[:2], dtype=np.int32)*np.arange(x_t.shape[1])[np.newaxis, :]
            x_inputs = [x_t, pos]
        else:
            raise ValueError("unknown input signature")

        model_out = model.predict(x_inputs, verbose=False)
        
        pred_y, pred_noise = None, None 
        if output_signature == "y,n":
            pred_y, pred_noise = model_out
        elif output_signature == "y":
            pred_y = model_out
        elif output_signature == "n":
            pred_noise = model_out
        else:
            raise ValueError("unknown output signature")
        
        if model_output_type == "continuous":
            continue #shouldn't need to do anything
        elif model_output_type == "bit-probability":
            if not pred_y is None:
                pred_y = bits2int(
                    np.random.random(pred_y.shape) < pred_y,
                    dtype=x_t.dtype,
                ).numpy()
            if not pred_noise is None:
                pred_noise = bits2int(
                    np.random.random(pred_noise.shape) < pred_noise,
                    dtype=x_t.dtype,
                ).numpy()
        elif model_output_type == "bit-logit":
            if not pred_y is None:
                pred_y = bits2int(
                    np.random.random(pred_y.shape) < sigmoid(pred_y),
                    dtype=x_t.dtype,
                ).numpy()
            if not pred_noise is None:
                pred_noise = bits2int(
                    np.random.random(pred_noise.shape) < sigmoid(pred_noise),
                    dtype=x_t.dtype,
                ).numpy()
        else:
            raise ValueError("unknown model_output_type")

        if estimate == "x-n":
            x_pred = x_t - pred_noise
        elif estimate == "y":
            x_pred = pred_y
        else:
            raise ValueError("unknown estimate method")
        
        x_t = update_fn(x_t, x_pred, gamma_now, gamma_next)
        
        if return_trajectory:
            trajectory.append(x_t)
    
    if not condition is None:
        #enforce the condition one more time after the generation process
        x_t = np.where(condition_mask_fn(x_t), x_t, condition)
    
    if return_trajectory:
        return x_t, trajectory
    
    return x_t
