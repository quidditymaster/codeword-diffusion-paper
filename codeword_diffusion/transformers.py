
import logging
import time

import numpy as np 

import tensorflow as tf 
import tensorflow.keras.layers as L

import codeword_diffusion as cdm

#resources
#https://www.tensorflow.org/text/tutorials/transformer

class BaseAttention(L.Layer):

    def __init__(self, **kwargs):
        super().__init__()
        self.mha = L.MultiHeadAttention(**kwargs)
        self.norm_x = L.LayerNormalization()
        self.add_l = L.Add()


#use the pre-norm formulation
#https://arxiv.org/pdf/2002.04745.pdf

class CrossAttention(BaseAttention):

    def call(self, x, context):
        att_out, attn_scores = self.mha(
            query=self.norm_x(x),
            key=context,
            value=context,
            return_attention_scores=True,
        )

        self.last_attn_scores = attn_scores
        
        x = self.add_l([x, att_out])
        return x

class GlobalSelfAttention(BaseAttention):

    def __init__(self, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, x, attention_mask=None):
        nx = self.norm_x(x)
        att_out = self.mha(
            query=nx,
            value=nx,
            key=nx,
            attention_mask=attention_mask,
        )
        if not self.alpha is None:
            alpha = self.alpha
            x = alpha*x + (1-alpha)*att_out
        else:
            x = x + att_out
        return x

class CausalSelfAttention(BaseAttention):

    def call(self, x):
        nx = self.norm_x(x)
        att_out = self.mha(
            query=nx,
            value=nx,
            key=nx,
            use_causal_mask=True,
        )
        x = self.add([x, att_out])
        return x

class FeedForward(L.Layer):

    def __init__(
        self, 
        d_model,
        d_squeeze, 
        dropout=None,
        alpha=None,
    ):
        super().__init__()
        self.dmid = L.Dense(d_squeeze, activation='relu')
        self.dout = L.Dense(d_model)
        self.dropout = None
        if not dropout is None:
            self.dropout = L.Dropout(dropout)
        self.norm_l = L.LayerNormalization()
        self.alpha = alpha

    def call(self, x):
        x_in = x
        x = self.norm_l(x)
        x = self.dmid(x)
        x = self.dout(x)
        if not self.dropout is None:
            x = self.dropout(x)
        if not self.alpha is None:
            alpha = self.alpha
            x = alpha*x_in + (1-alpha)*x
        else:
            x = x_in + x        
        return x


class EncoderBlock(L.Layer):

    def __init__(
        self, 
        d_model, 
        d_squeeze, 
        num_heads, 
        mha_dropout=None,
        ff_dropout=None,
        alpha=None,
    ):
        super().__init__()
        self.satt = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=mha_dropout,
            alpha=alpha,
        )
        self.ffn = FeedForward(
            d_model=d_model, 
            d_squeeze=d_squeeze, 
            dropout=ff_dropout,
            alpha=alpha,
        )

    def call(self, x, attention_mask=None):
        x = self.satt(x, attention_mask=attention_mask)
        x = self.ffn(x)
        return x


class KeySeparatedPositionEncoderBlock(L.Layer):

    def __init__(
        self, 
        d_model, 
        num_heads, 
        d_squeeze, 
        dropout=None,
    ):
        super().__init__()
        self.mha = L.MultiHeadAttention()
        self.norm_x = L.LayerNormalization()
        self.add_l = L.Add()

        self.satt = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            #should do something with dropout here?
        )
        self.ffn = FeedForward(
            d_model=d_model, 
            d_squeeze=d_squeeze, 
            dropout=dropout
        )

    def call(self, x):
        x = self.satt(x)
        x = self.ffn(x)
        return x


def make_fourier_basis_matrix(
    Lmax,
    n_channels,
    fmin=None,
    fmax=0.5,
    freq_dist='linear',
    dtype=np.float32
):
    if fmin is None:
        fmin = 0.5/Lmax

    assert fmin > 0

    n_freqs = n_channels//2

    if freq_dist == 'linear':
        freqs = np.linspace(fmin, fmax, n_freqs)
    elif freq_dist == 'log-linear':
        freqs = np.exp(np.linspace(np.log(fmin), np.log(fmax), n_freqs))
    else:
        raise ValueError("unknown value for freq_dist", freq_dist)

    freqs = freqs[np.newaxis]
    xv = np.arange(Lmax)[:, np.newaxis]

    radians = 2*np.pi*freqs*xv

    return np.concatenate(
        [
            np.cos(radians),
            np.sin(radians),
        ],
        axis=-1,
    ).astype(dtype)
    


def exponential_positional_encoding(
    length, 
    n_channels,
):
  n_freqs = n_channels//2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)


class PrecomputedPositionEmbedding(L.Layer):

    def __init__(
        self,
        pos_matrix,
    ):
        super().__init__()
        self.pos_matrix = pos_matrix
    
    def call(self, x):
        bsize = tf.shape(x)[0]
        #return tf.ones_like(x) * self.pos_matrix[tf.newaxis]
        return tf.repeat(self.pos_matrix[tf.newaxis], bsize, axis=0)


def build_model(
    Lmax, 
    bit_width,
    input_signature="x,p",
    input_type="int",
    embedding_type="lookup",
    cbe_kwargs=None,
    noise_dim=128,
    d_model=32,
    d_squeeze=64,
    num_heads=4,
    n_levels=4,
    mha_dropout=0.1,
    ff_dropout=0.1,
    mix_alpha=None,
    pred_activation=None,
    noise_activation=None,
):  
    if input_type == "int":
        x_in = L.Input([Lmax], dtype=tf.int32)
        if embedding_type == "lookup":
            x = L.Embedding(2**bit_width, d_model)(x_in)
        elif embedding_type == "circular-bitshift":
            cbe = cdm.CircularBitshiftEmbedding(**cbe_kwargs)
            x = cbe(x_in)
            x = L.Dense(d_model, activation=None)(x)
        else:
            raise ValueError("uknown embedding type")
    elif input_type == "float":
        x_in = L.Input([Lmax, bit_width])
        x = L.Dense(d_model)(x_in)

    if "m" in input_signature:
        mask_in = L.Input([Lmax, Lmax])

    xl0 = x
    
    pos_in = L.Input([Lmax], dtype=tf.int32)
    pos_emb = L.Embedding(Lmax, d_model)(pos_in)
    
    x = x+pos_emb

    for i in range(n_levels):
        x = EncoderBlock(
            d_model=d_model,
            d_squeeze=d_squeeze,
            num_heads=num_heads,
            ff_dropout=ff_dropout,
            mha_dropout=mha_dropout,
            alpha=mix_alpha,
        )(x, attention_mask=mask_in)
    
    x_pred = L.Dense(bit_width, activation=pred_activation)(x)

    x = L.Dense(noise_dim, activation="relu")(L.concatenate([x_pred, xl0, x]))
    x = L.Dense(noise_dim, activation="relu")(x)

    x_noise = L.Dense(bit_width, activation=noise_activation)(x)

    if "m" in input_signature:
        model = tf.keras.models.Model([x_in, pos_in, mask_in], [x_pred, x_noise])
    
    return model
