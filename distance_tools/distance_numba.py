from __future__ import print_function, absolute_import
from numba import cuda
import numpy
import math
import numba as nb
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

size = -1
k = -1
kplus1 = -1


# CUDA kplus1ernel : compute symmetric distance matrix between solutions of the same set for each pop - O(S) Daniel Approximation
@cuda.jit
def computeSymmetricMatrixDistance_PorumbelApprox( size_sub_pop, matrixDistance, tColor):

    d = cuda.grid(1)

    if (d < (size_sub_pop*(size_sub_pop-1)/2)):

        # Get upper triangular matrix indices from thread index !
        idx1 = int(size_sub_pop - 2 - int(math.sqrt(-8.0 * d + 4.0 * size_sub_pop * (size_sub_pop - 1) - 7) / 2.0 - 0.5))
        idx2 = int(d + idx1 + 1 - size_sub_pop * (size_sub_pop - 1) / 2 + (size_sub_pop - idx1) * ((size_sub_pop - idx1) - 1) / 2)

        ttNbSameColor = nb.cuda.local.array((kplus1, kplus1), nb.uint8)

        M = nb.cuda.local.array((kplus1), nb.int16)
        sigma = nb.cuda.local.array((kplus1), nb.int16)

        for j in range(kplus1):
            M[j] = 0
            sigma[j] = 0

        for x in range(size):
            ttNbSameColor[int(tColor[int(idx1), x]), int(tColor[int(idx2), x])] = 0

        for x in range(size):

            i = int(tColor[int(idx1), x])
            j = int(tColor[int(idx2), x])

            ttNbSameColor[i, j] += 1

            if(ttNbSameColor[i, j] > M[i]):
                M[i] = ttNbSameColor[i, j]
                sigma[i] = j

        proxi = 0
        for i in range(kplus1):
            proxi += ttNbSameColor[i, sigma[i]]


        matrixDistance[ int(idx1), int(idx2)] = size - proxi
        matrixDistance[ int(idx2), int(idx1)] = size - proxi




# CUDA kplus1ernel : compute distance matrix between two set of solutions for each pop
@cuda.jit
def computeMatrixDistance_PorumbelApprox(size_sub_pop, size_sub_pop2, matrixDistance, tColor1, tColor2):

    d = cuda.grid(1)


    if (d < size_sub_pop*size_sub_pop2):

        idx1 = int(d // size_sub_pop2)
        idx2 = int(d % size_sub_pop2)

        ttNbSameColor = nb.cuda.local.array((kplus1, kplus1), nb.uint8)

        M = nb.cuda.local.array((kplus1), nb.int16)
        sigma = nb.cuda.local.array((kplus1), nb.int16)

        for i in range(kplus1):
            M[i] = 0
            sigma[i] = 0

        for x in range(size):
            ttNbSameColor[int(tColor1[int(idx1), x]), int(tColor2[int(idx2), x])] = 0

        for x in range(size):

            i = int(tColor1[int(idx1), x])
            j = int(tColor2[int(idx2), x])

            ttNbSameColor[i, j] += 1

            if (ttNbSameColor[i, j] > M[i]):
                M[i] = ttNbSameColor[i, j]
                sigma[i] = j

        proxi = 0
        for i in range(kplus1):
            proxi += ttNbSameColor[i, sigma[i]]

        matrixDistance[int(idx1), int(idx2)] = size - proxi





