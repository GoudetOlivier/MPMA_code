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


# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_newtest(rng_states,  size_pop, size_sub_pop, A, nb_neighbors, tColor, fils, matrice_already_tested, closest_individuals,  fit_crossover):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        idx_in_pop = d


        bestIdx = idx_in_pop
        for w in range(nb_neighbors):

            e = int(closest_individuals[idx_in_pop,w])

            if(idx_in_pop!=e and matrice_already_tested[idx_in_pop,e]==0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[ e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = i

                            for l in range(nbParent):
                                if (parents[l, j] < k):
                                    tSizeOfColors[l, parents[l, j]] -= 1


                for j in range(size):

                    if (current_child[j] < 0):

                        r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                        if (r >= k):
                            r = k - 1

                        current_child[j] = r

                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1

                if(f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]


        fit_crossover[d] = bestFit

        matrice_already_tested[idx_in_pop, bestIdx] = 1
        
        


# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def compute_nearest_neighbor_crossovers_Latin_Square(rng_states, size_pop, V, L, tColor, fils,
                                 matrice_already_tested, closest_individuals):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        e = int(closest_individuals[d, 0])

        if (d != e):

            current_child = nb.cuda.local.array((size), nb.int16)
            parents = nb.cuda.local.array((nbParent, size), nb.int16)

            for j in range(size):
                parents[0, j] = tColor[d, j]
                parents[1, j] = tColor[ e, j]

            for j in range(size):
                current_child[j] = -1

            tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

            for i in range(nbParent):
                for j in range(k):
                    tSizeOfColors[i, j] = 0

                for j in range(size):
                    if (parents[i, j] < k):
                        tSizeOfColors[i, parents[i, j]] += 1


            for i in range(k):

                if(i%3 == 0 or i%3 == 1):
                    indiceParent = 0
                else:
                    indiceParent = 1

                valMax = -1
                colorMax = -1

                startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                for j in range(k):
                    color = (startColor + j) % k;
                    currentVal = tSizeOfColors[indiceParent, color]

                    if (currentVal > valMax):
                        valMax = currentVal
                        colorMax = color

                for j in range(size):
                    if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                        current_child[j] = colorMax

                        for l in range(nbParent):
                            if (parents[l, j] < k):
                                tSizeOfColors[l, parents[l, j]] -= 1

            for j in range(size):

                if (current_child[j] < 0):

                    nb_col = int(V[j])

                    r = int(nb_col * xoroshiro128p_uniform_float32(rng_states, d))

                    current_child[j] = int(L[j,r])


            for j in range(size):
                fils[d, j] = current_child[j]

        matrice_already_tested[d, e] = 1




# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def compute_nearest_neighbor_crossovers_Latin_Square_corrected(rng_states, size_pop, V, L, tColor, fils,
                                 matrice_already_tested, closest_individuals):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        e = int(closest_individuals[d, 0])

        if (d != e):

            current_child = nb.cuda.local.array((size), nb.int16)
            parents = nb.cuda.local.array((nbParent, size), nb.int16)

            for j in range(size):
                parents[0, j] = tColor[d, j]
                parents[1, j] = tColor[ e, j]

            for j in range(size):
                current_child[j] = -1

            tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

            for i in range(nbParent):
                for j in range(k):
                    tSizeOfColors[i, j] = 0

                for j in range(size):
                    if (parents[i, j] < k):
                        tSizeOfColors[i, parents[i, j]] += 1


            for i in range(k):

                if(i%3 == 0 or i%3 == 1):
                    indiceParent = 0
                else:
                    indiceParent = 1

                valMax = -1
                colorMax = -1

                startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                for j in range(k):
                    color = (startColor + j) % k;
                    currentVal = tSizeOfColors[indiceParent, color]

                    if (currentVal > valMax):
                        valMax = currentVal
                        colorMax = color

                for j in range(size):
                    if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                        current_child[j] = colorMax

                        for l in range(nbParent):
                            if (parents[l, j] < k):
                                tSizeOfColors[l, parents[l, j]] -= 1

                            tSizeOfColors[l,colorMax] = -1

            for j in range(size):

                if (current_child[j] < 0):

                    nb_col = int(V[j])

                    r = int(nb_col * xoroshiro128p_uniform_float32(rng_states, d))

                    current_child[j] = int(L[j,r])


            for j in range(size):
                fils[d, j] = current_child[j]

        matrice_already_tested[d, e] = 1



# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def compute_nearest_neighbor_crossovers_Latin_Square_partial(rng_states, size_pop, V, L, tColor, fils,
                                 matrice_already_tested, closest_individuals):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        e = int(closest_individuals[d, 0])

        if (d != e):

            current_child = nb.cuda.local.array((size), nb.int16)
            parents = nb.cuda.local.array((nbParent, size), nb.int16)

            for j in range(size):
                parents[0, j] = tColor[d, j]
                parents[1, j] = tColor[e, j]

            for j in range(size):
                current_child[j] = -1

            tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

            for i in range(nbParent):
                for j in range(k):
                    tSizeOfColors[i, j] = 0

                for j in range(size):
                    if (parents[i, j] < k):
                        tSizeOfColors[i, parents[i, j]] += 1


            for i in range(k):

                if(i%3 == 0 or i%3 == 1):
                    indiceParent = 0
                else:
                    indiceParent = 1

                valMax = -1
                colorMax = -1

                startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                for j in range(k):
                    color = (startColor + j) % k;
                    currentVal = tSizeOfColors[indiceParent, color]

                    if (currentVal > valMax):
                        valMax = currentVal
                        colorMax = color

                for j in range(size):
                    if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                        current_child[j] = colorMax

                        for l in range(nbParent):
                            if (parents[l, j] < k):
                                tSizeOfColors[l, parents[l, j]] -= 1

                            tSizeOfColors[l,colorMax] = -1

            for j in range(size):

                if (current_child[j] < 0):

                    current_child[j] = k


            for j in range(size):
                fils[d, j] = current_child[j]

        matrice_already_tested[d, e] = 1





# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def compute_nearest_neighbor_crossovers_GCP(rng_states, size_pop, tColor, fils,
                                 matrice_already_tested, closest_individuals):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        e = int(closest_individuals[d, 0])

        if (d != e):

            current_child = nb.cuda.local.array((size), nb.int16)
            parents = nb.cuda.local.array((nbParent, size), nb.int16)

            for j in range(size):
                parents[0, j] = tColor[d, j]
                parents[1, j] = tColor[ e, j]

            for j in range(size):
                current_child[j] = -1

            tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

            for i in range(nbParent):
                for j in range(k):
                    tSizeOfColors[i, j] = 0

                for j in range(size):
                    if (parents[i, j] < k):
                        tSizeOfColors[i, parents[i, j]] += 1


            for i in range(k):

                if(i%2 == 0):
                    indiceParent = 0
                else:
                    indiceParent = 1

                valMax = -1
                colorMax = -1

                startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                for j in range(k):
                    color = (startColor + j) % k;
                    currentVal = tSizeOfColors[indiceParent, color]

                    if (currentVal > valMax):
                        valMax = currentVal
                        colorMax = color

                for j in range(size):
                    if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                        current_child[j] = colorMax

                        for l in range(nbParent):
                            if (parents[l, j] < k):
                                tSizeOfColors[l, parents[l, j]] -= 1

            for j in range(size):

                if (current_child[j] < 0):

                    r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                    current_child[j] = r


            for j in range(size):
                fils[d, j] = current_child[j]

        matrice_already_tested[d, e] = 1





# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBest_Best_Unbalanced_Crossovers_KNN_Latin_Square(rng_states, size_pop, size_sub_pop, A, V, L, nb_neighbors, tColor, fils,
                                 matrice_already_tested, closest_individuals,  fit_crossover):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):



        bestIdx = d
        for w in range(nb_neighbors):

            e = int(closest_individuals[d, w])

            if (d != e and matrice_already_tested[d, e] == 0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[ e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1


                for i in range(k):

                    if(i%3 == 0 or i%3 == 1):
                        indiceParent = 0
                    else:
                        indiceParent = 1

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = colorMax

                            for l in range(nbParent):
                                if (parents[l, j] < k):
                                    tSizeOfColors[l, parents[l, j]] -= 1

                for j in range(size):

                    if (current_child[j] < 0):

                        nb_col = int(V[j])

                        r = int(nb_col * xoroshiro128p_uniform_float32(rng_states, d))

                        current_child[j] = int(L[j,r])

                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1

                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[d, bestIdx] = 1



# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBest_Distance_Unbalanced_Crossovers_KNN(rng_states, size_pop, size_sub_pop, A,  nb_neighbors, tColor, fils,
                                 matrice_already_tested, closest_individuals, type_cross,  fit_crossover, proba):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):



        bestIdx = d
        for w in range(nb_neighbors):

            e = int(closest_individuals[d, w])

            if (d != e and matrice_already_tested[d, e] == 0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[ e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1



                for i in range(k):

                    if(type_cross[d,w] == 0):

                       indiceParent = i % 2

                    elif(type_cross[d,w] == 1):
                       if(i%3 == 0 or i%3 == 1):
                           indiceParent = 0
                       else:
                           indiceParent = 1


                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = colorMax

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1

                f = 0

                for j in range(size):

                    if (current_child[j] < 0):

                        current_child[j] = k
                        f += 1

                                

                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[d, bestIdx] = 1



# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBest_Unbalanced_distance_Crossovers_KNN_Latin_Square(rng_states, size_pop, size_sub_pop, A, V, L, nb_neighbors, tColor, fils,
                                 matrice_already_tested, closest_individuals, type_cross,  fit_crossover, proba):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):



        bestIdx = d
        for w in range(nb_neighbors):

            e = int(closest_individuals[d, w])

            if (d != e and matrice_already_tested[d, e] == 0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[ e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1



                for i in range(k):

                    if(type_cross[d,w] == 0):

                       indiceParent = i % 2

                    elif(type_cross[d,w] == 1):
                       if(i%3 == 0 or i%3 == 1):
                           indiceParent = 0
                       else:
                           indiceParent = 1


                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = colorMax

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1

                for j in range(size):

                    if (current_child[j] < 0):

                        nb_col = int(V[j])

                        r = int(nb_col * xoroshiro128p_uniform_float32(rng_states, d))



                        current_child[j] = int(L[j,r])



                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1

                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[d, bestIdx] = 1




# CUDA kernel : compute all fitness crossovers between individuals in pop (we do not store all the crossovers as it uses too much memory).
@cuda.jit
def computeAllCrossovers_KNN_latin_square(rng_states, size_pop, nb_neighbors,  closest_individuals, tColor, allFit, matrice_already_tested):

    d = cuda.grid(1)
    nbParent = 2

    if (d < size_pop*nb_neighbors):


        idx1 = int(d // nb_neighbors)
        idx2 = int(d % nb_neighbors)


        nn = int(closest_individuals[idx1,idx2])


        if(idx1 != nn and matrice_already_tested[idx1,nn]==0):

            parents = nb.cuda.local.array((nbParent, size), nb.int16)
            current_child = nb.cuda.local.array((size), nb.int16)

            for j in range(size):
                parents[0, j] = tColor[idx1, j]
                parents[1, j] = tColor[nn, j]

            for j in range(size):
                current_child[j] = -1

            tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

            for i in range(nbParent):
                for j in range(k):
                    tSizeOfColors[i, j] = 0

                for j in range(size):
                    if(parents[i, j] > -1):
                        tSizeOfColors[i, parents[i, j]] += 1

            for i in range(k):

                indiceParent = i % 2

                valMax = -1
                colorMax = -1

                startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                for j in range(k):
                    color = (startColor + j) % k;
                    currentVal = tSizeOfColors[indiceParent, color]

                    if (currentVal > valMax):
                        valMax = currentVal
                        colorMax = color

                for j in range(size):
                    if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                        current_child[j] = i

                        for l in range(nbParent):
                            if (parents[l, j] > -1):
                                tSizeOfColors[l, parents[l, j]] -= 1

            f = 0
            for j in range(size):

                if (current_child[j] < 0):

                    f += 1



            allFit[int(idx1*size_pop + nn )] = f

        else:

            allFit[int(idx1 * size_pop + nn )] = 9999


# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBest_Unbalanced_Crossovers_KNN_Latin_Square(rng_states, size_pop, size_sub_pop, A, V, L, nb_neighbors, tColor, fils,
                                 matrice_already_tested, closest_individuals, fit_crossover, proba):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):



        bestIdx = d
        for w in range(nb_neighbors):

            e = int(closest_individuals[d, w])

            if (d != e and matrice_already_tested[d, e] == 0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[ e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1

                nbParentZero = int(k*proba)
                nbParentOne = k - nbParentZero

                for i in range(k):

                    r = xoroshiro128p_uniform_float32(rng_states, d)

                    if(nbParentZero ==0):
                        indiceParent = 1
                    elif(nbParentOne==0):
                        indiceParent = 0
                    else:
                        if(r < proba):
                            indiceParent = 0
                            nbParentZero-= 1
                        else:
                            indiceParent = 1
                            nbParentOne-= 1


                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = colorMax

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1

                f = 0

                for j in range(size):

                    if (current_child[j] < 0):


                        current_child[j] = k
                        f += 1


                                

                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[d, bestIdx] = 1




# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_Latin_Square_three_parents(rng_states,  size_pop, nb_neighbors, A,  V, L, tColor, fils, matrice_already_tested, closest_individuals, fit_crossover):

    d = cuda.grid(1)

    nbParent = 3

    bestFit = 9999

    if (d < size_pop):


        bestIdx1 = d
        bestIdx2 = d

        for w1 in range(nb_neighbors):

            e1 = int(closest_individuals[ d, w1])

            for w2 in range(nb_neighbors):

                e2 = int(closest_individuals[d,w2])

                if(d!=e1 and d!=e2 and e1!=e2 and matrice_already_tested[d,e1]==0 and matrice_already_tested[d,e2]==0):

                    current_child = nb.cuda.local.array((size), nb.int16)
                    parents = nb.cuda.local.array((nbParent, size), nb.int16)

                    for j in range(size):
                        parents[0, j] = tColor[d, j]
                        parents[1, j] = tColor[e1, j]
                        parents[2, j] = tColor[e2, j]

                    for j in range(size):
                        current_child[j] = -1

                    tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                    for i in range(nbParent):
                        for j in range(k):
                            tSizeOfColors[i, j] = 0

                        for j in range(size):
                            if(parents[i, j] > -1):
                                tSizeOfColors[i, parents[i, j]] += 1

                    for i in range(k):

                        indiceParent = i % 3

                        valMax = -1
                        colorMax = -1

                        startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                        for j in range(k):
                            color = (startColor + j) % k
                            currentVal = tSizeOfColors[indiceParent, color]

                            if (currentVal > valMax):
                                valMax = currentVal
                                colorMax = color

                        for j in range(size):
                            if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                                current_child[j] = colorMax

                                for l in range(nbParent):
                                    if (parents[l, j] > -1):
                                        tSizeOfColors[l, parents[l, j]] -= 1

                    for j in range(size):

                        if (current_child[j] < 0):

                            nb_col = int(V[j])

                            r = int(nb_col * xoroshiro128p_uniform_float32(rng_states, d))

     
                            current_child[j] = int(L[j,r])

                    f = 0
                    for x in range(size):
                        for y in range(x):
                            if (A[x, y] == 1):
                                if (current_child[x] == current_child[y]):
                                    f += 1


                    if(f < bestFit):
                        bestFit = f
                        bestIdx1 = e1
                        bestIdx2 = e2
                        for j in range(size):
                            fils[d, j] = current_child[j]
                            
        fit_crossover[d] = bestFit

        matrice_already_tested[ d, bestIdx1] = 1
        matrice_already_tested[d, bestIdx2] = 1



# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_Latin_Square_standardGPX(rng_states, size_pop, size_sub_pop, A, V, L, nb_neighbors, tColor, fils,
                                 matrice_already_tested, closest_individuals, fit_crossover):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):



        bestIdx = d
        for w in range(nb_neighbors):

            e = int(closest_individuals[d, w])

            if (d != e and matrice_already_tested[d, e] == 0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[ e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = colorMax

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1

                for j in range(size):

                    if (current_child[j] < 0):

                        nb_col = int(V[j])

                        r = int(nb_col * xoroshiro128p_uniform_float32(rng_states, d))



                        current_child[j] = int(L[j,r])



                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1


                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[d, bestIdx] = 1




# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_Latin_Square(rng_states, size_pop, size_sub_pop, A, V, L, nb_neighbors, tColor, fils,
                                 matrice_already_tested, closest_individuals, fit_crossover):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):



        bestIdx = d
        for w in range(nb_neighbors):

            e = int(closest_individuals[d, w])

            if (d != e and matrice_already_tested[d, e] == 0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[ e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = colorMax

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1

                f = 0

                for j in range(size):

                    if (current_child[j] < 0):


                        current_child[j] = k
                        f += 1

                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[d, bestIdx] = 1





# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_GreedyKNN_Latin_Square(rng_states, size_pop, size_sub_pop, A, V, L, nb_neighbors, tColor, fils,
                                 matrice_already_tested, closest_individuals, fit_crossover):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        idx_in_pop = d % size_sub_pop
        num_pop = d // size_sub_pop

        bestIdx = idx_in_pop
        for w in range(nb_neighbors):

            e = int(closest_individuals[num_pop, idx_in_pop, w])

            if (idx_in_pop != e and matrice_already_tested[num_pop, idx_in_pop, e] == 0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[num_pop * size_sub_pop + e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] > -1):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                    for j in range(k):
                        color = (startColor + j) % k
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = colorMax

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1



                f = 0
                startNode = int(size * xoroshiro128p_uniform_float32(rng_states, d))
                for j in range(size):
                    x = (startNode + j) % size

                    if (current_child[x] < 0):
                        minConflicts = 999
                        best_col = -1

                        nb_col = V[x]

                        for c in range(nb_col):

                            current_col = L[x,c]
                            nbConflicts = 0
                            for y in range(size):
                                if (A[x, y] == 1 and current_child[y] == current_col):
                                    nbConflicts += 1

                            if(nbConflicts < minConflicts):
                                minConflicts = nbConflicts
                                best_col = current_col

                        current_child[x] = best_col
                        f += minConflicts


                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[num_pop, idx_in_pop, bestIdx] = 1






# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestMAGX_KNN_Latin_Square(rng_states, size_pop, size_sub_pop, A, V, L, nb_neighbors, tColor, fils,
                                 matrice_already_tested, closest_individuals, fit_crossover):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        idx_in_pop = d % size_sub_pop
        num_pop = d // size_sub_pop

        bestIdx = idx_in_pop
        for w in range(nb_neighbors):

            e = int(closest_individuals[num_pop, idx_in_pop, w])

            if (idx_in_pop != e and matrice_already_tested[num_pop, idx_in_pop, e] == 0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[num_pop * size_sub_pop + e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] > -1):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    parMax = -1
                    valMax = -1
                    colorMax = -1


                    for indiceParent in range(nbParent):


                        startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                        for j in range(k):
                            color = (startColor + j) % k;
                            currentVal = tSizeOfColors[indiceParent, color]

                            if (currentVal > valMax):
                                valMax = currentVal
                                colorMax = color
                                parMax = indiceParent

                    for j in range(size):
                        if (parents[int(parMax), j] == colorMax and current_child[j] < 0):
                            current_child[j] = colorMax

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1


                for j in range(size):

                    if (current_child[j] < 0):

                        if(parents[0, j] == parents[1, j]):

                            current_child[j] = parents[0, j]



                for j in range(size):

                    if (current_child[j] < 0):

                        nb_col = int(V[j])

                        r = int(nb_col * xoroshiro128p_uniform_float32(rng_states, d))

                        # if (r >= nb_col):
                        #     r = nb_col - 1

                        current_child[j] = int(L[j,r])


                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1

                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[num_pop, idx_in_pop, bestIdx] = 1




# CUDA kernel : compute crossovers given a specific list of indices in pop
@cuda.jit
def computeSpecificCrossoversMSCP(rng_states, size_pop, A, tColor, allCrossovers, indices, matrice_already_tested):

    d = cuda.grid(1)
    nbParent = 2

    if (d < size_pop):

        idx1 = int(indices[d]//size_pop)
        idx2 = int(indices[d] %size_pop)

        matrice_already_tested[0,idx1,idx2] = 1

        parents = nb.cuda.local.array((nbParent, size), nb.int16)
        current_child = nb.cuda.local.array((size), nb.int16)

        for j in range(size):
            parents[0, j] = tColor[idx1, j]
            parents[1, j] = tColor[idx2, j]

        for j in range(size):
            current_child[j] = -1

        tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

        for i in range(nbParent):
            for j in range(k):
                tSizeOfColors[i, j] = 0

            for j in range(size):
                if(parents[i, j] > -1):
                    tSizeOfColors[i, parents[i, j]] += 1

        for i in range(k):

            indiceParent = i % 2

            valMax = -1
            colorMax = -1

            startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for j in range(k):
                color = (startColor + j) % k;
                currentVal = tSizeOfColors[indiceParent, color]

                if (currentVal > valMax):
                    valMax = currentVal
                    colorMax = color

            for j in range(size):
                if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                    current_child[j] = i

                    for l in range(nbParent):
                        if (parents[l, j] > -1):
                            tSizeOfColors[l, parents[l, j]] -= 1
        
        sumColor = 0
        for x in range(size):

            if (current_child[x] < 0):


                for l in range(k):


                    y = 0
                    found2 = False

                    for y in range(size):
                        if(A[x,y] == 1 and current_child[y] == l):
                            found2 = True
                            break					


                    if(found2 == False):
                        current_child[x] = l
                        break

                if (current_child[x] < 0):

                    current_child[x] = 0


            sumColor += current_child[x] + 1

        for j in range(size):
            allCrossovers[d,j] = current_child[j]


# CUDA kernel : compute crossovers given a specific list of indices in pop
@cuda.jit
def computeSpecificCrossoversMSCPV2(rng_states, size_pop, A, tColor, allCrossovers, indices, matrice_already_tested, current_k):
    d = cuda.grid(1)
    nbParent = 2

    if (d < size_pop):

        idx1 = int(indices[d] // size_pop)
        idx2 = int(indices[d] % size_pop)

        matrice_already_tested[0, idx1, idx2] = 1

        parents = nb.cuda.local.array((nbParent, size), nb.int16)
        current_child = nb.cuda.local.array((size), nb.int16)

        for j in range(size):
            parents[0, j] = tColor[idx1, j]
            parents[1, j] = tColor[idx2, j]

        for j in range(size):
            current_child[j] = -1

        tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

        for i in range(nbParent):
            for j in range(current_k):
                tSizeOfColors[i, j] = 0

            for j in range(size):
                if (parents[i, j] > -1):
                    tSizeOfColors[i, parents[i, j]] += 1

        for i in range(current_k):

            indiceParent = i % 2

            valMax = -1
            colorMax = -1

            startColor = int(current_k * xoroshiro128p_uniform_float32(rng_states, d))

            for j in range(current_k):
                color = (startColor + j) % current_k;
                currentVal = tSizeOfColors[indiceParent, color]

                if (currentVal > valMax):
                    valMax = currentVal
                    colorMax = color

            for j in range(size):
                if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                    current_child[j] = i

                    for l in range(nbParent):
                        if (parents[l, j] > -1):
                            tSizeOfColors[l, parents[l, j]] -= 1

        for j in range(size):

            if (current_child[j] < 0):

                r = int(current_k* xoroshiro128p_uniform_float32(rng_states, d))

                if (r >= current_k):
                    r =current_k - 1

                current_child[j] = r

        for j in range(size):
            allCrossovers[d, j] = current_child[j]




# CUDA kernel : compute all fitness crossovers between individuals in pop (we do not store all the crossovers as it uses too much memory).
@cuda.jit
def computeAllGreedyCrossovers_KNN_MSCP(rng_states, size_pop, nb_neighbors, A, closest_individuals, tColor, allFit):

    d = cuda.grid(1)
    nbParent = 2

    if (d < size_pop*nb_neighbors):


        idx1 = int(d // nb_neighbors)
        idx2 = int(d % nb_neighbors)


        nn = int(closest_individuals[0,idx1,idx2])

        parents = nb.cuda.local.array((nbParent, size), nb.int16)
        current_child = nb.cuda.local.array((size), nb.int16)

        for j in range(size):
            parents[0, j] = tColor[idx1, j]
            parents[1, j] = tColor[nn, j]

        for j in range(size):
            current_child[j] = -1

        tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

        for i in range(nbParent):
            for j in range(k):
                tSizeOfColors[i, j] = 0

            for j in range(size):
                if(parents[i, j] > -1):
                    tSizeOfColors[i, parents[i, j]] += 1

        for i in range(k):

            indiceParent = i % 2

            valMax = -1
            colorMax = -1

            startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for j in range(k):
                color = (startColor + j) % k;
                currentVal = tSizeOfColors[indiceParent, color]

                if (currentVal > valMax):
                    valMax = currentVal
                    colorMax = color

            for j in range(size):
                if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                    current_child[j] = i

                    for l in range(nbParent):
                        if (parents[l, j] > -1):
                            tSizeOfColors[l, parents[l, j]] -= 1


        sumColor = 0
        for x in range(size):

            if (current_child[x] < 0):


                for l in range(k):


                    y = 0
                    found2 = False

                    for y in range(size):
                        if(A[x,y] == 1 and current_child[y] == l):
                            found2 = True
                            break;					


                    if(found2 == False):
                        current_child[x] = l
                        break;

                if (current_child[x] < 0):

                    current_child[x] = 0


            sumColor += current_child[x] + 1

        allFit[int(idx1*size_pop + nn )] = sumColor







# CUDA kernel : compute all fitness crossovers between individuals in pop (we do not store all the crossovers as it uses too much memory).
@cuda.jit
def computeAllCrossovers_KNN_MSCP(rng_states, size_pop, nb_neighbors, A, closest_individuals, tColor, allFit, mu, current_k):

    d = cuda.grid(1)
    nbParent = 2

    if (d < size_pop*nb_neighbors):


        idx1 = int(d // nb_neighbors)
        idx2 = int(d % nb_neighbors)


        nn = int(closest_individuals[0,idx1,idx2])




        parents = nb.cuda.local.array((nbParent, size), nb.int16)
        current_child = nb.cuda.local.array((size), nb.int16)

        for j in range(size):
            parents[0, j] = tColor[idx1, j]
            parents[1, j] = tColor[nn, j]

        for j in range(size):
            current_child[j] = -1

        tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

        for i in range(nbParent):
            for j in range(current_k):
                tSizeOfColors[i, j] = 0

            for j in range(size):
                if(parents[i, j] > -1):
                    tSizeOfColors[i, parents[i, j]] += 1

        for i in range(current_k):

            indiceParent = i % 2

            valMax = -1
            colorMax = -1

            startColor = int(current_k * xoroshiro128p_uniform_float32(rng_states, d))

            for j in range(current_k):
                color = (startColor + j) % current_k;
                currentVal = tSizeOfColors[indiceParent, color]

                if (currentVal > valMax):
                    valMax = currentVal
                    colorMax = color

            for j in range(size):
                if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                    current_child[j] = i

                    for l in range(nbParent):
                        if (parents[l, j] > -1):
                            tSizeOfColors[l, parents[l, j]] -= 1


        for j in range(size):

            if (current_child[j] < 0):

                r = int(current_k* xoroshiro128p_uniform_float32(rng_states, d))

                if (r >= current_k):
                    r =current_k - 1

                current_child[j] = r

        f = 0
        for x in range(size):
            for y in range(x):
                if (A[x, y] == 1):
                    if (current_child[x] == current_child[y]):
                        f += 1

        sumColor = 0
        for x in range(size):
            sumColor += current_child[x] + 1

        score_global = mu*sumColor  + f


        allFit[int(idx1*size_pop + nn )] = score_global



# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossoversMSCP_KNN_three_parents(rng_states, size_pop, size_sub_pop, nb_neighbors, A, tColor, fils,
                                             closest_individuals, fit_crossover, mu):
    d = cuda.grid(1)

    nbParent = 3

    bestFit = 9999

    if (d < size_pop):

        idx_in_pop = d % size_sub_pop
        num_pop = d // size_sub_pop

        bestIdx1 = idx_in_pop
        # bestIdx2 = idx_in_pop

        for w1 in range(nb_neighbors):
            e1 = int(closest_individuals[num_pop, idx_in_pop, w1])
            for w2 in range(nb_neighbors):

                e2 = int(closest_individuals[num_pop, idx_in_pop, w2])

                if (idx_in_pop != e1 and idx_in_pop != e2 and e1 != e2):

                    current_child = nb.cuda.local.array((size), nb.int16)
                    parents = nb.cuda.local.array((nbParent, size), nb.int16)

                    for j in range(size):
                        parents[0, j] = tColor[d, j]
                        parents[1, j] = tColor[num_pop * size_sub_pop + e1, j]
                        parents[2, j] = tColor[num_pop * size_sub_pop + e2, j]

                    for j in range(size):
                        current_child[j] = -1

                    tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                    for i in range(nbParent):
                        for j in range(k):
                            tSizeOfColors[i, j] = 0

                        for j in range(size):
                            if (parents[i, j] > -1):
                                tSizeOfColors[i, parents[i, j]] += 1

                    for i in range(k):

                        indiceParent = i % 3

                        valMax = -1
                        colorMax = -1

                        startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                        for j in range(k):
                            color = (startColor + j) % k
                            currentVal = tSizeOfColors[indiceParent, color]

                            if (currentVal > valMax):
                                valMax = currentVal
                                colorMax = color

                        for j in range(size):
                            if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                                current_child[j] = i

                                for l in range(nbParent):
                                    if (parents[l, j] > -1):
                                        tSizeOfColors[l, parents[l, j]] -= 1

                    for j in range(size):

                        if (current_child[j] < 0):

                            r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                            if (r >= k):
                                r = k - 1

                            current_child[j] = r

                    f = 0
                    for x in range(size):
                        for y in range(x):
                            if (A[x, y] == 1):
                                if (current_child[x] == current_child[y]):
                                    f += 1

                    sumColor = 0
                    for x in range(size):
                        sumColor += current_child[x] + 1

                    score_global = sumColor + mu * f


                    if (score_global < bestFit):
                        bestFit = score_global
                        bestIdx1 = e1
                        for j in range(size):
                            fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        #matrice_already_tested[num_pop, idx_in_pop, bestIdx1] += 1



# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_MSCP(rng_states,  size_pop, size_sub_pop, nb_neighbors, A,  tColor, fils, matrice_already_tested, closest_individuals,  fit_crossover, mu):

    d = cuda.grid(1)

    nbParent = 2

    bestscore_global = 9999

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop

        bestIdx = idx_in_pop
        for w in range(nb_neighbors):

            e = int(closest_individuals[idx_in_pop,w])

            if(idx_in_pop!=e and matrice_already_tested[idx_in_pop,e]==0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[num_pop * size_sub_pop + e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] > -1):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = i

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1

                for j in range(size):

                    if (current_child[j] < 0):

                        r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                        if (r >= k):
                            r = k - 1

                        current_child[j] = r

                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1

                sumColor = 0
                for x in range(size):
                    sumColor += current_child[x] + 1

                score_global = sumColor + mu * f


                if (score_global < bestscore_global):
                    bestscore_global = score_global
                    bestIdx = e


        fit_crossover[d] = bestscore_global

        matrice_already_tested[idx_in_pop, bestIdx] = 1






# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_WIPX(rng_states,  size_pop, size_sub_pop, A, Deg, tColor, fils, matrice_already_tested, fit_crossover):

    d = cuda.grid(1)


    nbParent = 2

    bestScore = 9999

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop

        bestIdx = idx_in_pop
        for e in range(size_sub_pop):

            if(idx_in_pop!=e and matrice_already_tested[num_pop,idx_in_pop,e]==0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[num_pop * size_sub_pop + e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)
                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if(parents[i, j] > -1):
                            tSizeOfColors[i, parents[i, j]] += 1

                # tConflicts = nb.cuda.local.array((nbParent, k), nb.int16)
                # for i in range(nbParent):
                #     for j in range(k):
                #         tConflicts[i, j] = 0

                # for i in range(nbParent):
                #     for x in range(size):
                #         for y in range(x):
                #             if (A[x, y] == 1):
                #                 if (parents[i,x] == parents[i,y]):
                #                     tConflicts[i,parents[i,x]] += 1

                tDegree = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tDegree[i, j] = 0

                    for j in range(size):
                        if(parents[i,j] > -1):
                            tDegree[i, parents[i,j]] += Deg[j]

                total_score = 0

                for c in range(k):

                    valMax = -1
                    colorMax = -1
                    indexParentMax = -1

                    for p in range(nbParent):

                        for j in range(k):
                            #tConflicts[p, j]
                            currentVal =  tSizeOfColors[p,j] + tDegree[p,j]/(E*size)

                            if (currentVal > valMax):
                                valMax = currentVal
                                colorMax = j
                                indexParentMax = p

                    total_score += valMax

                    for j in range(size):
                        if (parents[indexParentMax, j] == colorMax and current_child[j] < 0):
                            current_child[j] = c

                            for l in range(nbParent):
                                if(parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1
                                    tDegree[l, parents[l, j]] -= Deg[j]
                                    #tConflicts[l, parents[l, j]] -=

                    # for p in range(nbParent):
                    #     for j in range(k):
                    #         tConflicts[p, j] = 0
                    #
                    # for p in range(nbParent):
                    #     for i in range(size):
                    #         for j in range(i):
                    #             if (A[i, j] == 1):
                    #                 if (parents[p, i] == parents[p, j]):
                    #                     tConflicts[p, parents[p, i]] += 1

                if (total_score < bestScore):
                    bestScore = total_score
                    bestIdx = e
                    for j in range(size):
                        if(current_child[j] > -1):
                            fils[d, j] = current_child[j]
                        else:
                            r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                            if (r >= k):
                                r = k - 1

                            fils[d, j] = r

        fit_crossover[d] = bestScore
        matrice_already_tested[num_pop, idx_in_pop, bestIdx] = 1


@cuda.jit
def computeBestCrossovers_WIPX_KNN(rng_states,  size_pop, size_sub_pop, nb_neighbors, A, Deg, tColor, fils, matrice_already_tested, closest_individuals, fit_crossover):

    d = cuda.grid(1)

    nbParent = 2

    bestScore = 9999

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop

        bestIdx = idx_in_pop
        for w in range(nb_neighbors):

            e = int(closest_individuals[num_pop,idx_in_pop,w])

            if(idx_in_pop!=e and matrice_already_tested[num_pop,idx_in_pop,e]==0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[num_pop * size_sub_pop + e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)
                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if(parents[i, j] > -1):
                            tSizeOfColors[i, parents[i, j]] += 1

                # tConflicts = nb.cuda.local.array((nbParent, k), nb.int16)
                # for i in range(nbParent):
                #     for j in range(k):
                #         tConflicts[i, j] = 0

                # for i in range(nbParent):
                #     for x in range(size):
                #         for y in range(x):
                #             if (A[x, y] == 1):
                #                 if (parents[i,x] == parents[i,y]):
                #                     tConflicts[i,parents[i,x]] += 1

                tDegree = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tDegree[i, j] = 0

                    for j in range(size):
                        if(parents[i,j] > -1):
                            tDegree[i, parents[i,j]] += Deg[j]

                total_score = 0

                for c in range(k):

                    valMax = -1
                    colorMax = -1
                    indexParentMax = -1

                    for p in range(nbParent):

                        for j in range(k):
                            #tConflicts[p, j]
                            currentVal =  tSizeOfColors[p,j] + tDegree[p,j]/(E*size)

                            if (currentVal > valMax):
                                valMax = currentVal
                                colorMax = j
                                indexParentMax = p

                    total_score += valMax

                    for j in range(size):
                        if (parents[indexParentMax, j] == colorMax and current_child[j] < 0):
                            current_child[j] = c

                            for l in range(nbParent):
                                if(parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1
                                    tDegree[l, parents[l, j]] -= Deg[j]
                                    #tConflicts[l, parents[l, j]] -=

                    # for p in range(nbParent):
                    #     for j in range(k):
                    #         tConflicts[p, j] = 0
                    #
                    # for p in range(nbParent):
                    #     for i in range(size):
                    #         for j in range(i):
                    #             if (A[i, j] == 1):
                    #                 if (parents[p, i] == parents[p, j]):
                    #                     tConflicts[p, parents[p, i]] += 1

                if (total_score < bestScore):
                    bestScore = total_score
                    bestIdx = e
                    for j in range(size):
                        if(current_child[j] > -1):
                            fils[d, j] = current_child[j]
                        else:
                            r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                            if (r >= k):
                                r = k - 1

                            fils[d, j] = r

        fit_crossover[d] = bestScore
        matrice_already_tested[num_pop, idx_in_pop, bestIdx] = 1




@cuda.jit
def computeBestCrossovers_WIPX_KNN_three_parents(rng_states,  size_pop, size_sub_pop, nb_neighbors, Deg, tColor, fils, closest_individuals, fit_crossover):

    d = cuda.grid(1)

    nbParent = 3

    bestScore = 9999

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop


        for w1 in range(nb_neighbors):
            e1 = int(closest_individuals[num_pop, idx_in_pop, w1])
            for w2 in range(nb_neighbors):

                e2 = int(closest_individuals[num_pop,idx_in_pop,w2])

                if(idx_in_pop!=e1 and idx_in_pop!=e2 and e1!=e2):

                    current_child = nb.cuda.local.array((size), nb.int16)
                    parents = nb.cuda.local.array((nbParent, size), nb.int16)

                    for j in range(size):
                        parents[0, j] = tColor[d, j]
                        parents[1, j] = tColor[num_pop * size_sub_pop + e1, j]
                        parents[2, j] = tColor[num_pop * size_sub_pop + e2, j]

                    for j in range(size):
                        current_child[j] = -1

                    tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)
                    for i in range(nbParent):
                        for j in range(k):
                            tSizeOfColors[i, j] = 0

                        for j in range(size):
                            if(parents[i, j] > -1):
                                tSizeOfColors[i, parents[i, j]] += 1

                    # tConflicts = nb.cuda.local.array((nbParent, k), nb.int16)
                    # for i in range(nbParent):
                    #     for j in range(k):
                    #         tConflicts[i, j] = 0

                    # for i in range(nbParent):
                    #     for x in range(size):
                    #         for y in range(x):
                    #             if (A[x, y] == 1):
                    #                 if (parents[i,x] == parents[i,y]):
                    #                     tConflicts[i,parents[i,x]] += 1

                    tDegree = nb.cuda.local.array((nbParent, k), nb.int16)

                    for i in range(nbParent):
                        for j in range(k):
                            tDegree[i, j] = 0

                        for j in range(size):
                            if(parents[i,j] > -1):
                                tDegree[i, parents[i,j]] += Deg[j]

                    total_score = 0

                    for c in range(k):

                        valMax = -1
                        colorMax = -1
                        indexParentMax = -1

                        for p in range(nbParent):

                            for j in range(k):
                                #tConflicts[p, j]
                                currentVal =  tSizeOfColors[p,j] + tDegree[p,j]/(E*size)

                                if (currentVal > valMax):
                                    valMax = currentVal
                                    colorMax = j
                                    indexParentMax = p

                        total_score += valMax

                        for j in range(size):
                            if (parents[indexParentMax, j] == colorMax and current_child[j] < 0):
                                current_child[j] = c

                                for l in range(nbParent):
                                    if(parents[l, j] > -1):
                                        tSizeOfColors[l, parents[l, j]] -= 1
                                        tDegree[l, parents[l, j]] -= Deg[j]
                                        #tConflicts[l, parents[l, j]] -=

                        # for p in range(nbParent):
                        #     for j in range(k):
                        #         tConflicts[p, j] = 0
                        #
                        # for p in range(nbParent):
                        #     for i in range(size):
                        #         for j in range(i):
                        #             if (A[i, j] == 1):
                        #                 if (parents[p, i] == parents[p, j]):
                        #                     tConflicts[p, parents[p, i]] += 1

                    if (total_score < bestScore):
                        bestScore = total_score

                        for j in range(size):
                            if(current_child[j] > -1):
                                fils[d, j] = current_child[j]
                            else:
                                r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                                if (r >= k):
                                    r = k - 1

                                fils[d, j] = r

        fit_crossover[d] = bestScore




# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_three_parents_withDeg(rng_states,  size_pop, size_sub_pop, nb_neighbors, Deg, tColor, fils, matrice_already_tested, closest_individuals, fit_crossover):

    d = cuda.grid(1)

    nbParent = 3

    bestFit = -1

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop

        bestIdx1 = idx_in_pop
        #bestIdx2 = idx_in_pop

        for w1 in range(nb_neighbors):

            
            e1 = int(closest_individuals[num_pop, idx_in_pop, w1])
            for w2 in range(nb_neighbors):

                e2 = int(closest_individuals[num_pop,idx_in_pop,w2])

                if(idx_in_pop!=e1 and idx_in_pop!=e2 and e1!=e2 and matrice_already_tested[num_pop,idx_in_pop,e1]<5):

                    f = 0

                    current_child = nb.cuda.local.array((size), nb.int16)
                    parents = nb.cuda.local.array((nbParent, size), nb.int16)

                    for j in range(size):
                        parents[0, j] = tColor[d, j]
                        parents[1, j] = tColor[num_pop * size_sub_pop + e1, j]
                        parents[2, j] = tColor[num_pop * size_sub_pop + e2, j]

                    for j in range(size):
                        current_child[j] = -1

                    tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                    for i in range(nbParent):
                        for j in range(k):
                            tSizeOfColors[i, j] = 0

                        for j in range(size):
                            if(parents[i, j] > -1):
                                tSizeOfColors[i, parents[i, j]] += 1

                    tDegree = nb.cuda.local.array((nbParent, k), nb.int16)

                    for i in range(nbParent):
                        for j in range(k):
                            tDegree[i, j] = 0

                        for j in range(size):
                            if(parents[i,j] > -1):
                                tDegree[i, parents[i,j]] += Deg[j]



                    for i in range(k):

                        indiceParent = i % 3

                        valMax = -1
                        colorMax = -1

                        #startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                        for color in range(k):
                            #color = (startColor + j) % k
                            currentVal = tSizeOfColors[indiceParent, color] + 10*tDegree[indiceParent,color]/(E*size)


                            if (currentVal > valMax):
                                valMax = currentVal
                                colorMax = color

                        f += valMax

                        for j in range(size):
                            if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                                current_child[j] = i

                                for l in range(nbParent):
                                    if (parents[l, j] > -1):
                                        tSizeOfColors[l, parents[l, j]] -= 1
                                        tDegree[l, parents[l, j]] -= Deg[j]


                    
                    if(f > bestFit):
                        bestFit = f
                        bestIdx1 = e1
                        for j in range(size):
                            if(current_child[j] > -1):
                                fils[d, j] = current_child[j]

        for j in range(size):
            if(fils[d, j] < 0):
                r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                if (r >= k):
                    r = k - 1

                fils[d, j] = r


        fit_crossover[d] = bestFit

        matrice_already_tested[num_pop, idx_in_pop, bestIdx1] += 1



# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_three_parents(rng_states,  size_pop, size_sub_pop, nb_neighbors, A, tColor, fils, matrice_already_tested, closest_individuals, fit_crossover):

    d = cuda.grid(1)

    nbParent = 3

    bestFit = 9999

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop

        bestIdx1 = idx_in_pop
        #bestIdx2 = idx_in_pop

        for w1 in range(nb_neighbors):
            e1 = int(closest_individuals[num_pop, idx_in_pop, w1])
            for w2 in range(nb_neighbors):

                e2 = int(closest_individuals[num_pop,idx_in_pop,w2])

                if(idx_in_pop!=e1 and idx_in_pop!=e2 and e1!=e2 and matrice_already_tested[num_pop,idx_in_pop,e1]==0):

                    current_child = nb.cuda.local.array((size), nb.int16)
                    parents = nb.cuda.local.array((nbParent, size), nb.int16)

                    for j in range(size):
                        parents[0, j] = tColor[d, j]
                        parents[1, j] = tColor[num_pop * size_sub_pop + e1, j]
                        parents[2, j] = tColor[num_pop * size_sub_pop + e2, j]

                    for j in range(size):
                        current_child[j] = -1

                    tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                    for i in range(nbParent):
                        for j in range(k):
                            tSizeOfColors[i, j] = 0

                        for j in range(size):
                            if(parents[i, j] > -1):
                                tSizeOfColors[i, parents[i, j]] += 1

                    for i in range(k):

                        indiceParent = i % 3

                        valMax = -1
                        colorMax = -1

                        startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                        for j in range(k):
                            color = (startColor + j) % k
                            currentVal = tSizeOfColors[indiceParent, color]

                            if (currentVal > valMax):
                                valMax = currentVal
                                colorMax = color

                        for j in range(size):
                            if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                                current_child[j] = i

                                for l in range(nbParent):
                                    if (parents[l, j] > -1):
                                        tSizeOfColors[l, parents[l, j]] -= 1

                    for j in range(size):

                        if (current_child[j] < 0):

                            r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                            if (r >= k):
                                r = k - 1

                            current_child[j] = r

                    f = 0
                    for x in range(size):
                        for y in range(x):
                            if (A[x, y] == 1):
                                if (current_child[x] == current_child[y]):
                                    f += 1


                    if(f < bestFit):
                        bestFit = f
                        bestIdx1 = e1
                        for j in range(size):
                            fils[d, j] = current_child[j]
                            
        fit_crossover[d] = bestFit

        matrice_already_tested[num_pop, idx_in_pop, bestIdx1] += 1




# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_equitable(rng_states,  size_pop, size_sub_pop, nb_neighbors, A, tColor, fils, matrice_already_tested, closest_individuals,  fit_crossover):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        bestIdx = d

        for w in range(nb_neighbors):

            e = int(closest_individuals[d,w])

            if(d!=e and matrice_already_tested[d,e]==0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMin = 9999
                    colorMin = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = abs(tSizeOfColors[indiceParent, color] - size/k)

                        if ( currentVal < valMin ):
                            valMin = currentVal
                            colorMin = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMin and current_child[j] < 0):
                            current_child[j] = i

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1


                for j in range(size):

                    if (current_child[j] < 0):

                        r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                        if (r >= k):
                                r = k - 1

                        current_child[j] = r

                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1

                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):            
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[d, bestIdx] = 1




# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN(rng_states,  size_pop, size_sub_pop, nb_neighbors, A, tColor, fils, matrice_already_tested, closest_individuals,  fit_crossover):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        bestIdx = d

        for w in range(nb_neighbors):

            e = int(closest_individuals[d,w])

            if(d!=e and matrice_already_tested[d,e]==0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = i

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1


                for j in range(size):

                    if (current_child[j] < 0):

                        r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                        if (r >= k):
                                r = k - 1

                        current_child[j] = r

                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1

                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[d, bestIdx] = 1




# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_partialCol(rng_states, size_pop, size_sub_pop, A, nb_neighbors, tColor, fils,
                                 matrice_already_tested, closest_individuals, fit_crossover):
    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        idx_in_pop = d % size_sub_pop
        num_pop = d // size_sub_pop

        bestIdx = idx_in_pop
        for w in range(nb_neighbors):

            e = int(closest_individuals[num_pop, idx_in_pop, w])

            if (idx_in_pop != e and matrice_already_tested[num_pop, idx_in_pop, e] == 0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[num_pop * size_sub_pop + e, j]

                for j in range(size):
                    current_child[j] = k

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                    for j in range(k):
                        color = (startColor + j) % k
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] == k):
                            current_child[j] = i

                            for l in range(nbParent):
                                if (parents[l, j] < k):
                                    tSizeOfColors[l, parents[l, j]] -= 1

                f = 0
                for j in range(size):
                    if (current_child[j] == k):
                        f += 1


                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[num_pop, idx_in_pop, bestIdx] = 1







# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_partialCol_Latin_Square(rng_states, size_pop, size_sub_pop, nb_neighbors, tColor, fils,
                                 matrice_already_tested, closest_individuals, fit_crossover):
    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        bestIdx = d

        for w in range(nb_neighbors):

            e = int(closest_individuals[ d, w])

            if (d != e and matrice_already_tested[ d, e] == 0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[e, j]

                for j in range(size):
                    current_child[j] = k

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] < k):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                    for j in range(k):
                        color = (startColor + j) % k
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] == k):
                            current_child[j] = colorMax

                            for l in range(nbParent):
                                if (parents[l, j] < k):
                                    tSizeOfColors[l, parents[l, j]] -= 1

                f = 0
                for j in range(size):
                    if (current_child[j] == k):
                        f += 1


                if (f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[ d, bestIdx] = 1




# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers_KNN_v2(rng_states,  size_pop, size_sub_pop, A, nb_neighbors, tColor, fils, matrice_already_tested, closest_individuals,  fit_crossover):

    d = cuda.grid(1)

    nbParent = 2

    bestFit = 9999

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop

        bestIdx = idx_in_pop
        for w in range(nb_neighbors):

            e = int(closest_individuals[num_pop,idx_in_pop,w])

            if(idx_in_pop!=e and matrice_already_tested[num_pop,idx_in_pop,e]==0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)

                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[num_pop * size_sub_pop + e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if (parents[i, j] > -1):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = i

                            for l in range(nbParent):
                                if (parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1


                for j in range(size):

                    if (current_child[j] < 0):

                        r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                        if (r >= k):
                            r = k - 1

                        current_child[j] = r

                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1

                if(f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]


        fit_crossover[d] = bestFit

        matrice_already_tested[num_pop, idx_in_pop, bestIdx] = 1
        
        

# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossovers(rng_states,  size_pop, size_sub_pop, A, tColor, fils, matrice_already_tested, fit_crossover):

    d = cuda.grid(1)


    nbParent = 2

    bestFit = 9999



    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop


        bestIdx = idx_in_pop
        for e in range(size_sub_pop):

            if(idx_in_pop!=e and matrice_already_tested[num_pop,idx_in_pop,e]==0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)


                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[num_pop * size_sub_pop + e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if(parents[i, j] > -1):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = i

                            for l in range(nbParent):
                                if(parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1


                for j in range(size):

                    if (current_child[j] < 0):

                        r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                        if (r >= k):
                            r = k - 1

                        current_child[j] = r

                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1

                if(f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_already_tested[num_pop, idx_in_pop, bestIdx] = 1


# CUDA kernel : compute crossovers given a specific list of indices in pop
@cuda.jit
def computeSpecificCrossovers(rng_states, size_pop,  tColor, allCrossovers, indices, matrice_already_tested):

    d = cuda.grid(1)
    nbParent = 2

    if (d < size_pop):

        idx1 = int(indices[d]//size_pop)
        idx2 = int(indices[d] %size_pop)

        matrice_already_tested[0,idx1,idx2] = 1

        parents = nb.cuda.local.array((nbParent, size), nb.int16)
        current_child = nb.cuda.local.array((size), nb.int16)

        for j in range(size):
            parents[0, j] = tColor[idx1, j]
            parents[1, j] = tColor[idx2, j]

        for j in range(size):
            current_child[j] = -1

        tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

        for i in range(nbParent):
            for j in range(k):
                tSizeOfColors[i, j] = 0

            for j in range(size):
                if(parents[i, j] > -1):
                    tSizeOfColors[i, parents[i, j]] += 1

        for i in range(k):

            indiceParent = i % 2

            valMax = -1
            colorMax = -1

            startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for j in range(k):
                color = (startColor + j) % k;
                currentVal = tSizeOfColors[indiceParent, color]

                if (currentVal > valMax):
                    valMax = currentVal
                    colorMax = color

            for j in range(size):
                if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                    current_child[j] = i

                    for l in range(nbParent):
                        if (parents[l, j] > -1):
                            tSizeOfColors[l, parents[l, j]] -= 1

        for j in range(size):

            if (current_child[j] < 0):

                r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                if (r >= k):
                    r = k - 1

                current_child[j] = r

        for j in range(size):
            allCrossovers[d,j] = current_child[j]



# CUDA kernel : compute crossovers given a specific list of indices in pop
@cuda.jit
def computeClosestCrossover(rng_states, size_pop, size_sub_pop, tColor, allCrossovers, indices):

    d = cuda.grid(1)
    nbParent = 2

    if (d < size_pop):

        idx_in_pop = d%size_sub_pop
        num_pop = d//size_sub_pop

        idx1 = int(idx_in_pop)
        idx2 = int(indices[num_pop,idx1])

        parents = nb.cuda.local.array((nbParent, size), nb.int16)
        current_child = nb.cuda.local.array((size), nb.int16)

        for j in range(size):
            parents[0, j] = tColor[idx1, j]
            parents[1, j] = tColor[idx2, j]

        for j in range(size):
            current_child[j] = -1

        tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

        for i in range(nbParent):
            for j in range(k):
                tSizeOfColors[i, j] = 0

            for j in range(size):
                if(parents[i, j] > -1):
                    tSizeOfColors[i, parents[i, j]] += 1

        for i in range(k):

            indiceParent = i % 2

            valMax = -1
            colorMax = -1

            startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for j in range(k):
                color = (startColor + j) % k;
                currentVal = tSizeOfColors[indiceParent, color]

                if (currentVal > valMax):
                    valMax = currentVal
                    colorMax = color

            for j in range(size):
                if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                    current_child[j] = i

                    for l in range(nbParent):
                        if (parents[l, j] > -1):
                            tSizeOfColors[l, parents[l, j]] -= 1

        for j in range(size):

            if (current_child[j] < 0):

                r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                if (r >= k):
                    r = k - 1

                current_child[j] = r

        for j in range(size):
            allCrossovers[d,j] = current_child[j]







# CUDA kernel : compute crossovers given a specific list of indices in pop
@cuda.jit
def computeHEADCrossover(rng_states, nb_head,  tColor, fils):

    d = cuda.grid(1)
    nbParent = 2

    if (d < 2*nb_head):

        if(d < nb_head):
            idx1 = int(d)
            idx2 = int(d + nb_head)
        else:
            idx1 = int(d%nb_head + nb_head)
            idx2 = int(d%nb_head)

        parents = nb.cuda.local.array((nbParent, size), nb.int16)
        current_child = nb.cuda.local.array((size), nb.int16)

        for j in range(size):
            parents[0, j] = tColor[idx1, j]
            parents[1, j] = tColor[idx2, j]

        for j in range(size):
            current_child[j] = -1

        tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

        for i in range(nbParent):
            for j in range(k):
                tSizeOfColors[i, j] = 0

            for j in range(size):
                if(parents[i, j] > -1):
                    tSizeOfColors[i, parents[i, j]] += 1

        for i in range(k):

            indiceParent = i % 2

            valMax = -1
            colorMax = -1

            startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for j in range(k):
                color = (startColor + j) % k;
                currentVal = tSizeOfColors[indiceParent, color]

                if (currentVal > valMax):
                    valMax = currentVal
                    colorMax = color

            for j in range(size):
                if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                    current_child[j] = i

                    for l in range(nbParent):
                        if (parents[l, j] > -1):
                            tSizeOfColors[l, parents[l, j]] -= 1

        for j in range(size):

            if (current_child[j] < 0):

                r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                if (r >= k):
                    r = k - 1

                current_child[j] = r

        for j in range(size):
            fils[d,j] = current_child[j]



# CUDA kernel : compute all fitness crossovers between individuals in pop (we do not store all the crossovers as it uses too much memory).
@cuda.jit
def computeAllCrossovers(rng_states, size_pop, A, tColor, allFit, matrice_already_tested):

    d = cuda.grid(1)
    nbParent = 2

    if (d < size_pop*size_pop):


        idx1 = int(d // size_pop)
        idx2 = int(d %size_pop)

        if(idx1 != idx2 and matrice_already_tested[0,idx1,idx2]==0):

            parents = nb.cuda.local.array((nbParent, size), nb.int16)
            current_child = nb.cuda.local.array((size), nb.int16)

            for j in range(size):
                parents[0, j] = tColor[idx1, j]
                parents[1, j] = tColor[idx2, j]

            for j in range(size):
                current_child[j] = -1

            tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

            for i in range(nbParent):
                for j in range(k):
                    tSizeOfColors[i, j] = 0

                for j in range(size):
                    if(parents[i, j] > -1):
                        tSizeOfColors[i, parents[i, j]] += 1

            for i in range(k):

                indiceParent = i % 2

                valMax = -1
                colorMax = -1

                startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                for j in range(k):
                    color = (startColor + j) % k;
                    currentVal = tSizeOfColors[indiceParent, color]

                    if (currentVal > valMax):
                        valMax = currentVal
                        colorMax = color

                for j in range(size):
                    if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                        current_child[j] = i

                        for l in range(nbParent):
                            if (parents[l, j] > -1):
                                tSizeOfColors[l, parents[l, j]] -= 1


            for j in range(size):

                if (current_child[j] < 0):

                    r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                    if (r >= k):
                        r = k - 1

                    current_child[j] = r

            f = 0
            for x in range(size):
                for y in range(x):
                    if (A[x, y] == 1):
                        if (current_child[x] == current_child[y]):
                            f += 1

            allFit[int(idx1*size_pop + idx2 )] = f

        else:

            allFit[int(idx1 * size_pop + idx2 )] = 9999



# CUDA kernel : compute all fitness crossovers between individuals in pop (we do not store all the crossovers as it uses too much memory).
@cuda.jit
def computeAllCrossovers_KNN(rng_states, size_pop, nb_neighbors, A, closest_individuals, tColor, allFit, matrice_already_tested):

    d = cuda.grid(1)
    nbParent = 2

    if (d < size_pop*nb_neighbors):


        idx1 = int(d // nb_neighbors)
        idx2 = int(d % nb_neighbors)


        nn = int(closest_individuals[0,idx1,idx2])


        if(idx1 != nn and matrice_already_tested[0,idx1,nn]==0):

            parents = nb.cuda.local.array((nbParent, size), nb.int16)
            current_child = nb.cuda.local.array((size), nb.int16)

            for j in range(size):
                parents[0, j] = tColor[idx1, j]
                parents[1, j] = tColor[nn, j]

            for j in range(size):
                current_child[j] = -1

            tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

            for i in range(nbParent):
                for j in range(k):
                    tSizeOfColors[i, j] = 0

                for j in range(size):
                    if(parents[i, j] > -1):
                        tSizeOfColors[i, parents[i, j]] += 1

            for i in range(k):

                indiceParent = i % 2

                valMax = -1
                colorMax = -1

                startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                for j in range(k):
                    color = (startColor + j) % k;
                    currentVal = tSizeOfColors[indiceParent, color]

                    if (currentVal > valMax):
                        valMax = currentVal
                        colorMax = color

                for j in range(size):
                    if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                        current_child[j] = i

                        for l in range(nbParent):
                            if (parents[l, j] > -1):
                                tSizeOfColors[l, parents[l, j]] -= 1


            for j in range(size):

                if (current_child[j] < 0):

                    r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                    if (r >= k):
                        r = k - 1

                    current_child[j] = r

            f = 0
            for x in range(size):
                for y in range(x):
                    if (A[x, y] == 1):
                        if (current_child[x] == current_child[y]):
                            f += 1

            allFit[int(idx1*size_pop + nn )] = f

        else:

            allFit[int(idx1 * size_pop + nn )] = 9999


# CUDA kernel : compute all crossovers between one indivual (idx) and all others individuals in pop
@cuda.jit
def computeAllPossibleCrossoversByIndividual(rng_states, idx, size_pop, tColor, allCrossover):

    d = cuda.grid(1)
    nbParent = 2

    if (d < size_pop):

        idx1 = idx
        idx2 = d

        # if(idx1!=idx2):

        parents = nb.cuda.local.array((nbParent, size), nb.int16)
        #current_child = nb.cuda.local.array((size), nb.int16)

        for j in range(size):
            parents[0, j] = tColor[idx1, j]
            parents[1, j] = tColor[idx2, j]

        for j in range(size):
            allCrossover[d,j] = -1

        tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

        for i in range(nbParent):
            for j in range(k):
                tSizeOfColors[i, j] = 0

            for j in range(size):
                if(parents[i, j] > -1):
                    tSizeOfColors[i, parents[i, j]] += 1

        for i in range(k):

            indiceParent = i % 2

            valMax = -1
            colorMax = -1

            startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

            for j in range(k):
                color = (startColor + j) % k;
                currentVal = tSizeOfColors[indiceParent, color]

                if (currentVal > valMax):
                    valMax = currentVal
                    colorMax = color

            for j in range(size):
                if (parents[int(indiceParent), j] == colorMax and allCrossover[d,j] < 0):
                    allCrossover[d,j] = i

                    for l in range(nbParent):
                        if(parents[l, j] > -1):
                            tSizeOfColors[l, parents[l, j]] -= 1

        for j in range(size):

            if (allCrossover[d,j] < 0):

                r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                if (r >= k):
                    r = k - 1

                allCrossover[d,j] = r
                
                
                
# CUDA kernel : compute best crossover for each individual according to fitness criterion
@cuda.jit
def computeBestCrossoversWithDist(rng_states,  size_pop, size_sub_pop, nb_neighbors, A, tColor, fils, matrix_neighbors, matrice_crossovers_already_tested, fit_crossover):

    d = cuda.grid(1)


    nbParent = 2

    bestFit = 9999


    num_pop = 0

    if (d < size_pop):


        bestIdx = d
        for w in range(nb_neighbors):
        
            e = matrix_neighbors[d,w]

            if(d!=e and matrice_crossovers_already_tested[num_pop,d,e]==0):

                current_child = nb.cuda.local.array((size), nb.int16)
                parents = nb.cuda.local.array((nbParent, size), nb.int16)


                for j in range(size):
                    parents[0, j] = tColor[d, j]
                    parents[1, j] = tColor[e, j]

                for j in range(size):
                    current_child[j] = -1

                tSizeOfColors = nb.cuda.local.array((nbParent, k), nb.int16)

                for i in range(nbParent):
                    for j in range(k):
                        tSizeOfColors[i, j] = 0

                    for j in range(size):
                        if(parents[i, j] > -1):
                            tSizeOfColors[i, parents[i, j]] += 1

                for i in range(k):

                    indiceParent = i % 2

                    valMax = -1
                    colorMax = -1

                    startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d));

                    for j in range(k):
                        color = (startColor + j) % k;
                        currentVal = tSizeOfColors[indiceParent, color]

                        if (currentVal > valMax):
                            valMax = currentVal
                            colorMax = color

                    for j in range(size):
                        if (parents[int(indiceParent), j] == colorMax and current_child[j] < 0):
                            current_child[j] = i

                            for l in range(nbParent):
                                if(parents[l, j] > -1):
                                    tSizeOfColors[l, parents[l, j]] -= 1


                for j in range(size):

                    if (current_child[j] < 0):

                        r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                        if (r >= k):
                            r = k - 1

                        current_child[j] = r

                f = 0
                for x in range(size):
                    for y in range(x):
                        if (A[x, y] == 1):
                            if (current_child[x] == current_child[y]):
                                f += 1

                if(f < bestFit):
                    bestFit = f
                    bestIdx = e
                    for j in range(size):
                        fils[d, j] = current_child[j]

        fit_crossover[d] = bestFit

        matrice_crossovers_already_tested[num_pop, d, bestIdx] = 1

