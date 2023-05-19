import os, sys

module_path=os.path.split(os.path.split(__file__)[0])[0]
print(module_path)
sys.path.append(module_path)

import numpy as np

import codeword_diffusion as cdm
import codeword_diffusion.datasets as cdmd


def test_batching():
    values = np.arange(10)
    sampler = cdmd.PermutationSampler(
        values
    )

    batches = []
    for i in range(3):
        batches.append(sampler.get_batch(7))
    perm_data = np.hstack(batches)[:20].reshape((2, 10))
    print(perm_data)
    assert len(np.unique(perm_data[0])) == 10
    assert len(np.unique(perm_data[1])) == 10
    