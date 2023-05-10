import numpy as np

from . import text2uint8


ri = np.arange(81)//9
ci = np.arange(81)%9
bi = (ri//3)*3 + ci//3
#a permutation that makes the 3x3 boxes have contiguous flat indexes 
_boxperm = np.argsort(bi)

def count_logic_violations(
    board,
):    
    runique = [len(np.unique(v)) for v in board.reshape((9, 9))]
    cunique = [len(np.unique(v)) for v in board.transpose().reshape((9, 9))]
    bunique = [len(np.unique(v)) for v in board[_boxperm].reshape((9, 9))]
    
    col_violations = 81-np.sum(runique)
    row_violations = 81-np.sum(cunique)
    box_violations = 81-np.sum(bunique)
    
    return col_violations, row_violations, box_violations

def count_vocabulary_violations(
    board_ints,
    vocabulary_set,
):
    return sum([v in vocabulary_set for v in board_ints])

def violation_score(board):
    vvs = count_vocabulary_violations
    lvs = sum(count_logic_violations(board_ints))


def build_model(
    Lmax, 
    bit_width,
    input_type="int",
    d_model=32,
    d_squeeze=64,
    num_heads=4,
    n_levels=4,
    dropout=-1.0,
    pred_activation=None,
    noise_activation=None,
):  
    if input_type == "int":
        x_in = L.Input([Lmax], dtype=tf.int32)
        x = L.Embedding(2**bit_width, d_model)(x_in)
    elif input_type == "float":
        x_in = L.Input([Lmax, bit_width])
        x = L.Dense(d_model)(x_in)
    
    pos_in = L.Input([Lmax], dtype=tf.int32)
    pos_emb = L.Embedding(Lmax, d_model)(pos_in)
    
    x = x+pos_emb

    for i in range(n_levels):
        x = bft.EncoderBlock(
            d_model=d_model,
            d_squeeze=d_squeeze,
            num_heads=num_heads,
            dropout=-1,#0.1,
            alpha=0.8,
        )(x)
    
    x_pred = L.Dense(bit_width, activation=pred_activation)(x)

    x = L.Dense(d_model, L.concatenate([x_pred, x_in, x]), activation=None)
    x = EncoderBlock(d_model=d_model, d_squeeze=d_squeeze, num_heads=num_heads, alpha=0.3)

    x_noise = L.Dense(bit_width, activation=noise_activation)(x)

    model = tf.keras.models.Model([x_in, pos_in], [x_pred, x_noise])
    
    return model
