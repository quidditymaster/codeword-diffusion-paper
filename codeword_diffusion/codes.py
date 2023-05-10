
import numpy as np
import itertools

from sklearn.neighbors import NearestNeighbors


from . import int2bits, bits2int


def gen_random_codebook(
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


def gen_kofn_codebook(n, k):
    "generates a code in which exaxtly k of n bits are hot in each vector"
    code_bits = []
    for indexes in itertools.combinations(np.arange(n), k):
        codeword = np.zeros(n)
        for i in indexes:
            codeword[i] = 1
        code_bits.append(codeword)
    code_bits = np.array(code_bits).astype(np.int32)
    cwds = cdm.bits2int(code_bits).numpy()
    return cwds

def gen_hamming_codebook(
    n_bits,
    as_ints=True,
):
    assert n_bits < 20 # lets not get crazy
    assert 2**int(np.floor(np.log2(n_bits))) == n_bits
    
    n_checks = int(np.floor(np.log2(n_bits)))
    k = n_bits-n_checks-1
    
    #import pdb; pdb.set_trace()
    
    messages = np.arange(2**k).astype(np.int32)
    mbits = cdm.int2bits(messages, n_bits=k)
    address_bits = cdm.int2bits(
        np.arange(n_bits).astype(np.int32), 
        n_bits=n_checks
    ).numpy()
        
    encoded_cols = np.zeros((2**k, n_bits), dtype=np.int32)
    
    mcidx = 0
    for col in range(1, n_bits):
        if np.sum(address_bits[col]) > 1:
            #print(col)
            encoded_cols[:, col] = mbits[:, mcidx]
            mcidx += 1
    
    for bit_idx in range(n_checks):
        covered_columns = np.where(address_bits[:, bit_idx])[0]
        parity = np.zeros(len(mbits))
        for c2 in covered_columns:
            parity += encoded_cols[:, c2]
        parity = parity % 2
        encoded_cols[:, 2**bit_idx] = parity
    
    #then finally do the global parity check
    encoded_cols[:, 0] = np.sum(encoded_cols, axis=1) % 2
    
    if as_ints:
        return cdm.bits2int(encoded_cols).numpy()
    else:
        encoded_cols



def make_corrector_lookup(
    codewords, 
    n_bits,
    dtype=_np_int_dtype,
):
    code_bits = int2bits(codewords, n_bits=n_bits).numpy()
    
    all_msgs = np.arange(2**n_bits).astype(dtype)
    all_msgs_bits = int2bits(all_msgs, n_bits=n_bits).numpy()

    nnsearcher = NearestNeighbors().fit(code_bits)
    n_search = min(len(codewords), n_bits+1)
    self_dists, _ = nnsearcher.kneighbors(code_bits, n_neighbors=n_search)
    
    dists, neighbor_indexes = nnsearcher.kneighbors(all_msgs_bits, n_neighbors=n_search)
    corrector_lookup = codewords[neighbor_indexes[:, 0]]
    
    info = dict(
        dists=dists,
        neighbor_indexes=neighbor_indexes,
        nnsearcher=nnsearcher,
        self_dists=self_dists,
    )
    
    return corrector_lookup, info