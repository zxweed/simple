import numpy as np
from numba import cuda


@cuda.jit
def vPIN_kernel(T, result, period, page_size):
    page_index = cuda.grid(1)
    page_start = period + page_index * page_size

    # during init stage we can't calculate anything, but cumulate the values
    A = B = 0
    for i in range(page_start-period, page_start):
        if T.Size[i] < 0:
            A += -T.Size[i]
        else:
            B += T.Size[i]

    for i in range(page_start, page_start + page_size, 1):
        if T.Size[i] < 0:
            A += -T.Size[i]
        else:
            B += T.Size[i]

        k = i - period
        if T.Size[k] < 0:
            A -= -T.Size[k]
        else:
            B -= T.Size[k]

        if i < T.size:
            result[i] = (B - A) / (B + A) * 100
