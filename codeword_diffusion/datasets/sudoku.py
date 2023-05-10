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
