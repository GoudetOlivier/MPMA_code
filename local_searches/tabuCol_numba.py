from __future__ import print_function, absolute_import
from numba import cuda
import numpy
import math
import numba as nb
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

size = -1
k = -1
E = -1

@cuda.jit
def MixcolNoRandom_LatinSquareNotAffectedNodes(rng_states,  D, max_iter, A,  L, vect_nb_col,  tColor,  vect_fit, tabuTenure, alpha, phi):

    d = cuda.grid(1)


    if (d < D):

        nbEmptyNode = 0
        nbConflitcs = 0

        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size, kplus1), nb.int8)

        for x in range(size):
            for y in range(k+1):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])
            if (tColor_local[x] == k):
                nbEmptyNode += 1

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1 and tColor_local[x] > -1 and tColor_local[y] > -1):
                    if (tColor_local[y] < k):
                        gamma[x, tColor_local[y]] += 1
                    if (tColor_local[x] < k):
                        gamma[y, tColor_local[x]] += 1

                        if (tColor_local[x] == tColor_local[y]):
                            nbConflitcs += 1

        f = nbConflitcs + phi*nbEmptyNode

        f_best = f

        best_delta_empty = 999

        #start_node = int(size * xoroshiro128p_uniform_float32(rng_states, d))

        for iter in range(max_iter):

            best_delta = 9999.0


            best_x = -1
            best_v = -1

            nbcfl = 0

            for x in range(size):

                #x = (start_node+w)%size

                if(tColor_local[x] > - 1):

                    if tColor_local[x] == k or gamma[x, tColor_local[x]] > 0:

                        if(gamma[x, tColor_local[x]] > 0):
                            nbcfl += 1

                        v_x = tColor_local[x]

                        nb_col = int(vect_nb_col[x])

                        for j in range(nb_col+1):

                            v = int(L[x, j])
                            delta_empty = 0

                            if(j == nb_col):
                                v = k
                                delta_empty = 1

                            if(v_x == k):
                                delta_empty = -1


                            delta_conflicts = gamma[x, v] - gamma[x, tColor_local[x]]

                            delt = delta_conflicts + phi*delta_empty


                            if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                                if (delt < best_delta):
                                    best_x = x
                                    best_v = v

                                    best_delta = delt
                                    best_delta_empty = delta_empty



            f += best_delta
            nbEmptyNode += best_delta_empty

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1 and tColor_local[y] > -1):
                    if(old_value < k):
                        gamma[y, old_value] -= 1
                    if(best_v < k):
                        gamma[y, best_v] += 1

            tColor_local[best_x] = best_v

            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * (nbcfl+ nbEmptyNode))
            tabuTenure[ d, best_x, old_value] = res + iter

            if (f < f_best):

                f_best = f

                for a in range(size):
                    tColor[d, a] = tColor_local[a]

        vect_fit[d] = f_best
 



@cuda.jit
def MixcolNoRandom_LatinSquare(rng_states,  D, max_iter, A,  L, vect_nb_col,  tColor,  vect_fit, tabuTenure, alpha, phi):

    d = cuda.grid(1)


    if (d < D):

        nbEmptyNode = 0
        nbConflitcs = 0

        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size, kplus1), nb.int8)

        for x in range(size):
            for y in range(k+1):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])
            if (tColor_local[x] == k):
                nbEmptyNode += 1

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):
                    if (tColor_local[y] < k):
                        gamma[x, tColor_local[y]] += 1
                    if (tColor_local[x] < k):
                        gamma[y, tColor_local[x]] += 1

                        if (tColor_local[x] == tColor_local[y]):
                            nbConflitcs += 1

        f = nbConflitcs + phi*nbEmptyNode

        f_best = f

        best_delta_empty = 999

        #start_node = int(size * xoroshiro128p_uniform_float32(rng_states, d))

        for iter in range(max_iter):

            best_delta = 9999.0


            best_x = -1
            best_v = -1

            nbcfl = 0

            for x in range(size):

                #x = (start_node+w)%size

                if tColor_local[x] == k or gamma[x, tColor_local[x]] > 0:

                    if(gamma[x, tColor_local[x]] > 0):
                        nbcfl += 1

                    v_x = tColor_local[x]

                    nb_col = int(vect_nb_col[x])

                    for j in range(nb_col+1):

                        v = int(L[x, j])
                        delta_empty = 0

                        if(j == nb_col):
                            v = k
                            delta_empty = 1

                        if(v_x == k):
                            delta_empty = -1


                        delta_conflicts = gamma[x, v] - gamma[x, tColor_local[x]]

                        delt = delta_conflicts + phi*delta_empty


                        if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):
                                best_x = x
                                best_v = v

                                best_delta = delt
                                best_delta_empty = delta_empty


            f += best_delta
            nbEmptyNode += best_delta_empty

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1):
                    if(old_value < k):
                        gamma[y, old_value] -= 1
                    if(best_v < k):
                        gamma[y, best_v] += 1

            tColor_local[best_x] = best_v

            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * (nbcfl+ nbEmptyNode))
            tabuTenure[ d, best_x, old_value] = res + iter

            if (f < f_best):

                f_best = f

                for a in range(size):
                    tColor[d, a] = tColor_local[a]

        vect_fit[d] = f_best






@cuda.jit
def Mixcol_LatinSquare(rng_states,  D, max_iter, A,  L, vect_nb_col,  tColor,  vect_fit, tabuTenure, alpha, phi):

    d = cuda.grid(1)


    if (d < D):

        nbEmptyNode = 0
        nbConflitcs = 0

        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size, kplus1), nb.int8)

        for x in range(size):
            for y in range(k+1):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])
            if (tColor_local[x] == k):
                nbEmptyNode += 1

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):
                    if (tColor_local[y] < k):
                        gamma[x, tColor_local[y]] += 1
                    if (tColor_local[x] < k):
                        gamma[y, tColor_local[x]] += 1

                        if (tColor_local[x] == tColor_local[y]):
                            nbConflitcs += 1

        f = nbConflitcs + phi*nbEmptyNode

        f_best = f

        best_delta_empty = 999

        #start_node = int(size * xoroshiro128p_uniform_float32(rng_states, d))

        for iter in range(max_iter):

            best_delta = 9999.0


            best_x = -1
            best_v = -1

            nbcfl = 0

            trouve = 1

            #start_node = int(size*xoroshiro128p_uniform_float32(rng_states, d))

            for x in range(size):

                #x = (start_node+w)%size

                if tColor_local[x] == k or gamma[x, tColor_local[x]] > 0:

                    if(gamma[x, tColor_local[x]] > 0):
                        nbcfl += 1

                    v_x = tColor_local[x]

                    nb_col = int(vect_nb_col[x])

                    #start_color = int((nb_col + 1)*xoroshiro128p_uniform_float32(rng_states, d))

                    for j in range(nb_col+1):

                        #j = (start_color +c)%(nb_col + 1)

                        v = int(L[x, j])
                        delta_empty = 0

                        if(j == nb_col):
                            v = k
                            delta_empty = 1

                        if(v_x == k):
                            delta_empty = -1


                        delta_conflicts = gamma[x, v] - gamma[x, tColor_local[x]]

                        delt = delta_conflicts + phi*delta_empty




                        if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):
                                best_x = x
                                best_v = v

                                best_delta = delt
                                best_delta_empty = delta_empty
                                
                                trouve = 1

                            elif(delt == best_delta):

                                trouve += 1

                                if(int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):

                                    best_x = x
                                    best_v = v
                                    best_delta = delt
                                    best_delta_empty = delta_empty




            f += best_delta
            nbEmptyNode += best_delta_empty

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1):
                    if(old_value < k):
                        gamma[y, old_value] -= 1
                    if(best_v < k):
                        gamma[y, best_v] += 1

            tColor_local[best_x] = best_v

            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * (nbcfl+ nbEmptyNode))
            tabuTenure[ d, best_x, old_value] = res + iter

            if (f < f_best):

                f_best = f

                for a in range(size):
                    tColor[d, a] = tColor_local[a]

        vect_fit[d] = f_best





@cuda.jit
def MixcolNoRandom_LatinSquare_bigSizeGraph(rng_states,  D, max_iter, A,  L, vect_nb_col,  tColor,  vect_fit, tabuTenure, alpha, phi, gamma):

    d = cuda.grid(1)


    if (d < D):

        nbEmptyNode = 0
        nbConflitcs = 0

        tColor_local = nb.cuda.local.array((size), nb.int8)


        for x in range(size):
            for y in range(k+1):
                gamma[d, x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])
            if (tColor_local[x] == k):
                nbEmptyNode += 1

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):
                    if (tColor_local[y] < k):
                        gamma[d, x, tColor_local[y]] += 1
                    if (tColor_local[x] < k):
                        gamma[d, y, tColor_local[x]] += 1

                        if (tColor_local[x] == tColor_local[y]):
                            nbConflitcs += 1

        f = nbConflitcs + phi*nbEmptyNode

        f_best = f

        best_delta_empty = 999

        #start_node = int(size * xoroshiro128p_uniform_float32(rng_states, d))

        for iter in range(max_iter):

            best_delta = 9999.0


            best_x = -1
            best_v = -1

            nbcfl = 0

            for x in range(size):

                #x = (start_node+w)%size

                if tColor_local[x] == k or gamma[d, x, tColor_local[x]] > 0:

                    if(gamma[d, x, tColor_local[x]] > 0):
                        nbcfl += 1

                    v_x = tColor_local[x]

                    nb_col = int(vect_nb_col[x])

                    for j in range(nb_col+1):

                        v = int(L[x, j])
                        delta_empty = 0

                        if(j == nb_col):
                            v = k
                            delta_empty = 1

                        if(v_x == k):
                            delta_empty = -1


                        delta_conflicts = gamma[d, x, v] - gamma[d, x, tColor_local[x]]

                        delt = delta_conflicts + phi*delta_empty


                        if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):
                                best_x = x
                                best_v = v

                                best_delta = delt
                                best_delta_empty = delta_empty


            f += best_delta
            nbEmptyNode += best_delta_empty

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1):
                    if(old_value < k):
                        gamma[d, y, old_value] -= 1
                    if(best_v < k):
                        gamma[d, y, best_v] += 1

            tColor_local[best_x] = best_v

            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * (nbcfl+ nbEmptyNode))
            tabuTenure[ d, best_x, old_value] = res + iter

            if (f < f_best):

                f_best = f

                for a in range(size):
                    tColor[d, a] = tColor_local[a]

        vect_fit[d] = f_best



# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucolNoRandom(rng_states,  D, max_iter, A, tColor,  vect_fit, tabuTenure, alpha):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    gamma[x, tColor_local[y]] += 1
                    gamma[y, tColor_local[x]] += 1

                    if (tColor_local[x] == tColor_local[y]):
                        f += 1

        f_best = f



        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0

            # start_node = int(size*xoroshiro128p_uniform_float32(rng_states, d))
            # start_color = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for x in range(size):

                # x = (start_node+w)%size

                if gamma[x, tColor_local[x]] > 0:

                    nbcfl += 1

                    v_x = tColor_local[x]

                    for v in range(k):

                        # v = (start_color+c)%k

                        delt = gamma[x, v] - gamma[x, tColor_local[x]]

                        if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):
                                best_x = x
                                best_v = v

                                best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1):
                    gamma[y, old_value] -= 1
                    gamma[y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl) 
            tabuTenure[ d, best_x, old_value] = res + iter


            if (f <= f_best):
                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best



# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucolNoRandom_NotAffectedNode(rng_states,  D, max_iter, A, tColor,  vect_fit, tabuTenure, alpha):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1 ):

                    if(tColor_local[y] > -1):
                        gamma[x, tColor_local[y]] += 1

                    if( tColor_local[x] > -1):
                        gamma[y, tColor_local[x]] += 1

                        if (tColor_local[x] == tColor_local[y]):
                            f += 1

        f_best = f



        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0

            # start_node = int(size*xoroshiro128p_uniform_float32(rng_states, d))
            # start_color = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for x in range(size):

                # x = (start_node+w)%size

                if( tColor_local[x] > -1):

                    if gamma[x, tColor_local[x]] > 0:

                        nbcfl += 1

                        v_x = tColor_local[x]

                        for v in range(k):

                            # v = (start_color+c)%k

                            delt = gamma[x, v] - gamma[x, tColor_local[x]]

                            if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                                if (delt < best_delta):
                                    best_x = x
                                    best_v = v

                                    best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1 and tColor_local[y] > -1):
                    gamma[y, old_value] -= 1
                    gamma[y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter


            if (f <= f_best):
                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best


# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucolNoRandom_LatinSquareNotAffectedNode(rng_states,  D, max_iter, A, L, vect_nb_col, tColor,  vect_fit, tabuTenure, alpha):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    if(tColor_local[x] > -1):
                        gamma[y, tColor_local[x]] += 1
                    if(tColor_local[y] > -1):
                        gamma[x, tColor_local[y]] += 1
                    
                        if (tColor_local[x] == tColor_local[y]):
                            f += 1

        f_best = f


        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0

            # start_node = int(size*xoroshiro128p_uniform_float32(rng_states, d))
            # start_color = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for x in range(size):

                # x = (start_node+w)%size

                if(tColor_local[x] > -1):

                    if gamma[x, tColor_local[x]] > 0:

                        nbcfl += 1

                        v_x = tColor_local[x]

                        nb_col = int(vect_nb_col[x])

                        for j in range(nb_col):

                            v = int(L[x,j])

                            # v = (start_color+c)%k

                            delt = gamma[x, v] - gamma[x, tColor_local[x]]

                            if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                                if (delt < best_delta):
                                    best_x = x
                                    best_v = v

                                    best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1 and  tColor_local[y] > -1):
                    gamma[y, old_value] -= 1
                    gamma[y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter

            if (f <= f_best):

                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best




# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucol_LatinSquareNotAffectedNode(rng_states,  D, max_iter, A, L, vect_nb_col, tColor,  vect_fit, tabuTenure, alpha):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    if(tColor_local[x] > -1):
                        gamma[y, tColor_local[x]] += 1
                    if(tColor_local[y] > -1):
                        gamma[x, tColor_local[y]] += 1
                    
                        if (tColor_local[x] == tColor_local[y]):
                            f += 1

        f_best = f


        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0

            trouve = 1

            for x in range(size):


                if(tColor_local[x] > -1):

                    if gamma[x, tColor_local[x]] > 0:

                        nbcfl += 1

                        v_x = tColor_local[x]

                        nb_col = int(vect_nb_col[x])


                        for j in range(nb_col):



                            v = int(L[x,j])


                            delt = gamma[x, v] - gamma[x, tColor_local[x]]

                            if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                                if (delt < best_delta):
                                    best_x = x
                                    best_v = v

                                    best_delta = delt
                                    trouve  = 1

                            elif(delt == best_delta):

                                trouve += 1

                                if(int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):

                                    best_x = x
                                    best_v = v
                                    best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1 and  tColor_local[y] > -1):
                    gamma[y, old_value] -= 1
                    gamma[y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter

            if (f <= f_best):

                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best




# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucolNoRandom_LatinSquareNotAffectedNode_bigSizeGraph(rng_states,  D, max_iter, A, L, vect_nb_col, tColor,  vect_fit, tabuTenure, alpha, gamma ):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        #gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[d, x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    if(tColor_local[x] > -1):
                        gamma[d, y, tColor_local[x]] += 1
                    if(tColor_local[y] > -1):
                        gamma[d, x, tColor_local[y]] += 1
                    
                        if (tColor_local[x] == tColor_local[y]):
                            f += 1

        f_best = f


        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0

            # start_node = int(size*xoroshiro128p_uniform_float32(rng_states, d))
            # start_color = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for x in range(size):

                # x = (start_node+w)%size

                if(tColor_local[x] > -1):

                    if gamma[d, x, tColor_local[x]] > 0:

                        nbcfl += 1

                        v_x = tColor_local[x]

                        nb_col = int(vect_nb_col[x])

                        for j in range(nb_col):

                            v = int(L[x,j])

                            # v = (start_color+c)%k

                            delt = gamma[d, x, v] - gamma[d, x, tColor_local[x]]

                            if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                                if (delt < best_delta):
                                    best_x = x
                                    best_v = v

                                    best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1 and  tColor_local[y] > -1):
                    gamma[d, y, old_value] -= 1
                    gamma[d, y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter

            if (f <= f_best):

                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best



# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucolNoRandom_LatinSquare(rng_states,  D, max_iter, A, L, vect_nb_col, tColor,  vect_fit, tabuTenure, alpha):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    gamma[x, tColor_local[y]] += 1
                    gamma[y, tColor_local[x]] += 1

                    if (tColor_local[x] == tColor_local[y]):
                        f += 1

        f_best = f




        f_has_changed = 0

        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0

            # start_node = int(size*xoroshiro128p_uniform_float32(rng_states, d))
            # start_color = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for x in range(size):

                # x = (start_node+w)%size

                if gamma[x, tColor_local[x]] > 0:

                    nbcfl += 1

                    v_x = tColor_local[x]

                    nb_col = int(vect_nb_col[x])

                    for j in range(nb_col):

                        v = int(L[x,j])

                        # v = (start_color+c)%k

                        delt = gamma[x, v] - gamma[x, tColor_local[x]]

                        if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):
                                best_x = x
                                best_v = v

                                best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1):
                    gamma[y, old_value] -= 1
                    gamma[y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter

            f_has_changed += 1
            if (f <= f_best):
                f_has_changed = 0
                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best









# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def partialcol_LatinSquare(rng_states,  D, max_iter, A,  L, vect_nb_col, tColor,  vect_fit, tabuTenure, alpha):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])
            if(tColor_local[x] == k):
                f += 1

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):
                    if( tColor_local[y] < k):
                        gamma[x, tColor_local[y]] += 1
                    if( tColor_local[x] < k):
                        gamma[y, tColor_local[x]] += 1



        f_best = f


        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            trouve = 1

            for x in range(size):

                if tColor_local[x] == k:

                    nb_col = int(vect_nb_col[x])

                    for j in range(nb_col):

                        v = int(L[x,j])

                        delt = gamma[x, v] - 1

                        if ( (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):

                                best_x = x
                                best_v = v
                                best_delta = delt
                                
                                trouve = 1

                            elif(delt == best_delta):

                                trouve += 1

                                if(int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):

                                    best_x = x
                                    best_v = v
                                    best_delta = delt




            f += best_delta


            for y in range(size):

                if (A[best_x, y] == 1):
                    gamma[y, best_v] += 1

                    if(tColor_local[y] == best_v):

                        tColor_local[y] = k
                        for z in range(size):
                            if(A[y, z] == 1):
                                gamma[z, best_v] -= 1

                        tabuTenure[d, y, best_v] = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * f)



            tColor_local[best_x] = best_v


            if (f < f_best):

                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best



# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def partialcol_LatinSquareNotAffectedNode(rng_states,  D, max_iter, A,  L, vect_nb_col, tColor,  vect_fit, tabuTenure, alpha):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])
            if(tColor_local[x] == k):
                f += 1

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1 and tColor_local[y] > -1 and tColor_local[x] > -1):
                    if( tColor_local[y] < k):
                        gamma[x, tColor_local[y]] += 1
                    if( tColor_local[x] < k):
                        gamma[y, tColor_local[x]] += 1



        f_best = f


        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            trouve = 1

            for x in range(size):

                if tColor_local[x] == k :

                    nb_col = int(vect_nb_col[x])

                    for j in range(nb_col):

                        v = int(L[x,j])

                        delt = gamma[x, v] - 1

                        if ( (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):

                                best_x = x
                                best_v = v
                                best_delta = delt
                                
                                trouve = 1

                            elif(delt == best_delta):

                                trouve += 1

                                if(int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):

                                    best_x = x
                                    best_v = v
                                    best_delta = delt




            f += best_delta


            for y in range(size):

                if (A[best_x, y] == 1 and tColor_local[y] > -1):

                    gamma[y, best_v] += 1

                    if(tColor_local[y] == best_v):

                        tColor_local[y] = k
                        for z in range(size):
                            if(A[y, z] == 1 and tColor_local[z] > -1):
                                gamma[z, best_v] -= 1

                        tabuTenure[d, y, best_v] = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * f)



            tColor_local[best_x] = best_v


            if (f < f_best):

                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best






# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucol_LatinSquare(rng_states,  D, max_iter, A, L, vect_nb_col, tColor,  vect_fit, tabuTenure, alpha):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    gamma[x, tColor_local[y]] += 1
                    gamma[y, tColor_local[x]] += 1

                    if (tColor_local[x] == tColor_local[y]):
                        f += 1

        f_best = f




        f_has_changed = 0

        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0
            
            trouve = 1
            
            for x in range(size):


                if gamma[x, tColor_local[x]] > 0:

                    nbcfl += 1

                    v_x = tColor_local[x]

                    nb_col = int(vect_nb_col[x])

                    for j in range(nb_col):

                        v = int(L[x,j])

                        delt = gamma[x, v] - gamma[x, tColor_local[x]]

                        if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):
                                best_x = x
                                best_v = v

                                best_delta = delt
                                trouve  = 1

                            elif(delt == best_delta):

                                trouve += 1

                                if(int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):

                                    best_x = x
                                    best_v = v
                                    best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1):
                    gamma[y, old_value] -= 1
                    gamma[y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter

            f_has_changed += 1
            if (f <= f_best):
                f_has_changed = 0
                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best




# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucol_LatinSquare_BigSize(rng_states,  D, max_iter, A, L, vect_nb_col, tColor,  vect_fit, tabuTenure, alpha, gamma ):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int16)
        

        for x in range(size):
            for y in range(k):
                gamma[d, x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    gamma[d, x, tColor_local[y]] += 1
                    gamma[d, y, tColor_local[x]] += 1

                    if (tColor_local[x] == tColor_local[y]):
                        f += 1

        f_best = f




        f_has_changed = 0

        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0

            
            trouve = 1
            
            for x in range(size):

                if gamma[d, x, tColor_local[x]] > 0:

                    nbcfl += 1

                    v_x = tColor_local[x]

                    nb_col = int(vect_nb_col[x])

                    for j in range(nb_col):

                        v = int(L[x,j])

                        
                        delt = gamma[d, x, v] - gamma[d, x, tColor_local[x]]

                        if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):
                                best_x = x
                                best_v = v

                                best_delta = delt
                                
                                trouve = 1
                                
                            elif(delt == best_delta):

                                trouve += 1

                                if(int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):

                                    best_x = x
                                    best_v = v
                                    best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1):
                    gamma[d, y, old_value] -= 1
                    gamma[d, y, best_v] += 1

            tColor_local[best_x] = best_v;

            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter

            f_has_changed += 1
            if (f <= f_best):
                f_has_changed = 0
                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best



# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucol_LatinSquareNotAffectedNode_BigSize(rng_states,  D, max_iter, A, L, vect_nb_col, tColor,  vect_fit, tabuTenure, alpha, gamma):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[d, x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    if(tColor_local[x] > -1):
                        gamma[d, y, tColor_local[x]] += 1
                    if(tColor_local[y] > -1):
                        gamma[d, x, tColor_local[y]] += 1
                    
                        if (tColor_local[x] == tColor_local[y]):
                            f += 1

        f_best = f


        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0

            trouve = 1

            for x in range(size):


                if(tColor_local[x] > -1):

                    if gamma[d, x, tColor_local[x]] > 0:

                        nbcfl += 1

                        v_x = tColor_local[x]

                        nb_col = int(vect_nb_col[x])


                        for j in range(nb_col):



                            v = int(L[x,j])


                            delt = gamma[d, x, v] - gamma[d, x, tColor_local[x]]

                            if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                                if (delt < best_delta):
                                    best_x = x
                                    best_v = v

                                    best_delta = delt
                                    
                                    trouve = 1

                            elif(delt == best_delta):

                                trouve += 1

                                if(int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):

                                    best_x = x
                                    best_v = v
                                    best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1 and  tColor_local[y] > -1):
                    gamma[d, y, old_value] -= 1
                    gamma[d, y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter

            if (f <= f_best):

                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best



# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucolNoRandom_LatinSquare_bigSizeGraph(rng_states,  D, max_iter, A, L, vect_nb_col, tColor,  vect_fit, tabuTenure, alpha, gamma):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[d, x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    gamma[d, x, tColor_local[y]] += 1
                    gamma[d, y, tColor_local[x]] += 1

                    if (tColor_local[x] == tColor_local[y]):
                        f += 1

        f_best = f




        f_has_changed = 0

        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0

            # start_node = int(size*xoroshiro128p_uniform_float32(rng_states, d))
            # start_color = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for x in range(size):

                # x = (start_node+w)%size

                if gamma[d, x, tColor_local[x]] > 0:

                    nbcfl += 1

                    v_x = tColor_local[x]

                    nb_col = int(vect_nb_col[x])

                    for j in range(nb_col):

                        v = int(L[x,j])

                        # v = (start_color+c)%k

                        delt = gamma[d, x, v] - gamma[d, x, tColor_local[x]]

                        if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):
                                best_x = x
                                best_v = v

                                best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1):
                    gamma[d, y, old_value] -= 1
                    gamma[d, y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter

            f_has_changed += 1
            if (f <= f_best):
                f_has_changed = 0
                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best





# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucol(rng_states,  D, max_iter, A, tColor,  vect_fit, tabuTenure, alpha):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    gamma[x, tColor_local[y]] += 1
                    gamma[y, tColor_local[x]] += 1

                    if (tColor_local[x] == tColor_local[y]):
                        f += 1

        f_best = f




        f_has_changed = 0

        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0
            
            trouve = 1
            
            for x in range(size):


                if gamma[x, tColor_local[x]] > 0:

                    nbcfl += 1

                    v_x = tColor_local[x]

                    for v in range(k):

                        delt = gamma[x, v] - gamma[x, tColor_local[x]]

                        if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                            if (delt < best_delta):
                                best_x = x
                                best_v = v

                                best_delta = delt
                                
                                trouve = 1
                                
                            elif(delt == best_delta):

                                trouve += 1

                                if(int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):

                                    best_x = x
                                    best_v = v
                                    best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1):
                    gamma[y, old_value] -= 1
                    gamma[y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter

            f_has_changed += 1
            if (f <= f_best):
                f_has_changed = 0
                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best





# Tabu CUDA kernel : launch D tabucol in parallel on GPU without collecting examples
@cuda.jit
def tabucol_NotAffectedNode(rng_states,  D, max_iter, A, tColor,  vect_fit, tabuTenure, alpha):


    d = cuda.grid(1)

    if (d < D):

        f = 0


        tColor_local = nb.cuda.local.array((size), nb.int8)
        gamma = nb.cuda.local.array((size ,k), nb.int8)


        for x in range(size):
            for y in range(k):
                gamma[x, y] = 0
                tabuTenure[d, x, y] = 0

        for x in range(size):
            tColor_local[x] = int(tColor[d, x])

        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):

                    if(tColor_local[x] > -1):
                        gamma[y, tColor_local[x]] += 1
                    if(tColor_local[y] > -1):
                        gamma[x, tColor_local[y]] += 1
                    
                        if (tColor_local[x] == tColor_local[y]):
                            f += 1

        f_best = f


        for iter in range(max_iter):

            best_delta = 9999
            best_x = -1
            best_v = -1

            nbcfl = 0

            trouve = 1

            for x in range(size):


                if(tColor_local[x] > -1):

                    if gamma[x, tColor_local[x]] > 0:

                        nbcfl += 1

                        v_x = tColor_local[x]


                        for v in range(k):

                            delt = gamma[x, v] - gamma[x, tColor_local[x]]

                            if (v != v_x and (tabuTenure[d, x, v] <= iter or delt + f < f_best)):

                                if (delt < best_delta):
                                    best_x = x
                                    best_v = v

                                    best_delta = delt
                                    
                                    trouve = 1

                            elif(delt == best_delta):

                                trouve += 1

                                if(int(trouve * xoroshiro128p_uniform_float32(rng_states, d)) == 0):

                                    best_x = x
                                    best_v = v
                                    best_delta = delt



            f += best_delta

            old_value = tColor_local[best_x]

            for y in range(size):

                if (A[best_x, y] == 1 and  tColor_local[y] > -1):
                    gamma[y, old_value] -= 1
                    gamma[y, best_v] += 1

            tColor_local[best_x] = best_v;
            res = int(10 * xoroshiro128p_uniform_float32(rng_states, d)) + int(alpha * nbcfl)
            tabuTenure[ d, best_x, old_value] = res + iter

            if (f <= f_best):

                f_best = f
                for a in range(size):
                    tColor[d, a] = tColor_local[a]



        vect_fit[d] = f_best


