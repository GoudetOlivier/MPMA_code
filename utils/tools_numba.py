from __future__ import print_function, absolute_import
from numba import cuda
import numpy
import math
import numba as nb
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

size = -1
k = -1
size_pop = -1
E = -1


# Cuda kernel allowing to remove conflicts of D solutions in parallel
@cuda.jit
def greedyRemoveConflicts(rng_states, D, A, tColor):

    d = cuda.grid(1)

    if (d < D):

        tConflicts = nb.cuda.local.array((size), nb.int8)

        for x in range(size):
            tConflicts[x] = 0

        for x in range(size):
            for y in range(x):
                if(A[x,y]==1):
                    if ( tColor[d,x] == tColor[d,y]):
                        tConflicts[x] += 1
                        tConflicts[y] += 1

        nbNodes_conflicts = 0

        for x in range(size):
            if(tConflicts[x] > 0):
                nbNodes_conflicts += 1

        while(nbNodes_conflicts > 0):

            idx_max = -1
            valMaxConflict = -1

            startNode = int(size * xoroshiro128p_uniform_float32(rng_states, d));

            for i in range(size):
                x = (startNode + i) % size
                if(tConflicts[x] > valMaxConflict ):
                    valMaxConflict = tConflicts[x]
                    idx_max = x

            tConflicts[idx_max] = 0
            nbNodes_conflicts -= 1  

            for y in range(size):

                if(A[idx_max,y]==1):

                    if (tColor[d,idx_max] == tColor[d,y]):

                        if(tConflicts[y] > 0):

                            tConflicts[y] -= 1

                            if(tConflicts[y] == 0):
                                nbNodes_conflicts -= 1


            tColor[d,idx_max] = k





@cuda.jit
def initPop( rng_states, D, tColor, L, V):

    d = cuda.grid(1)

    if (d < D):

        for i in range(size):

            nb_col = int(V[i])
            r = int(nb_col * xoroshiro128p_uniform_float32(rng_states, d))
            tColor[d,i] = int(L[i, r])


@cuda.jit
def initPopGCP( rng_states, D, tColor):

    d = cuda.grid(1)

    if (d < D):

        for i in range(size):

            r = int(k * xoroshiro128p_uniform_float32(rng_states, d))
            tColor[d,i] = r



@cuda.jit
def fileNodeNoColor( rng_states, D, tColor, L, V):

    d = cuda.grid(1)

    if (d < D):

        for i in range(size):
            if (tColor[d,i] < 0):

                nb_col = int(V[i])

                r = int(nb_col * xoroshiro128p_uniform_float32(rng_states, d))

                tColor[d,i] = int(L[i, r])


def preprocess_PLSE_instance(filepath, nameGraph, maxCol, kernalization):

    print("Preprocess PLSE instance")
    
    ####Load edges
    filename = filepath + nameGraph

    n = maxCol*maxCol
    triangsup = np.zeros((n, n), dtype = int)

    listcols = np.zeros((n, maxCol+1))

    f = open(filename, "r")
    for line in f:
        x = line.split(sep = " ")
        # print(x)
        if(x[0] == 'e'):
            triangsup[int(x[1]) - 1, int(x[2]) - 1] = 1

        if (x[0] == 'f'):
            cpt = 0
            for i in range(1,len(x)):
                if(x[i] != '\n'):
                    listcols[int(x[1])-1,cpt] = int(x[i] )
                cpt += 1


    A = triangsup + np.transpose(triangsup)

    if(np.sum(listcols) == 0):
        for i in range(n):
            for j in range(maxCol):
                listcols[i,j+1] = j + 1
    
    dico_D = {}

    nbNodeswithOneColor = 0

    list_nodes_one_color = []

    for i in range(n):

        list_color_vertex = []

        nbcolor = 0

        for j in range(maxCol):

            if(listcols[i,j+1] > 0):
                list_color_vertex.append(listcols[i,j+1]-1)
                nbcolor += 1

        if(nbcolor == 1):
            nbNodeswithOneColor += 1
            list_nodes_one_color.append(i)

        if(nbcolor != 1 and nbcolor != maxCol):
            print("PBBBB")


        dico_D[i] = list_color_vertex


    list_removed_node = []
    list_affected_colors = []


    if(kernalization):

        nb_node_removed_kernalization = 0


        while (nbNodeswithOneColor > 0):

            for i in range(n):

                if (i not in list_removed_node):
                    D = dico_D[i]

                    if (len(D) == 1):

                        list_affected_colors.append(D[0])
                        dico_D.pop(i)
                        nbNodeswithOneColor -= 1
                        list_removed_node.append(i)

                        for j in range(n):

                            if (A[i, j] == 1 and j not in list_removed_node):

                                list_ = dico_D[j]
                                if (D[0] in list_):
                                    list_.remove(D[0])

                                    dico_D[j] = list_

                                    if (len(list_) == 1):
                                        nb_node_removed_kernalization += 1
                                        nbNodeswithOneColor += 1

        print("nb node removed kernalization")
        print(nb_node_removed_kernalization)

    else:

        for i in range(n):

            if(i not in list_removed_node):

                D = dico_D[i]


                if(len(D) == 1):

                    color = D[0]

                    list_affected_colors.append(color)

                    list_removed_node.append(i)

                    for j in range(n):

                        if(A[i,j] == 1 and (j not in list_removed_node) ):


                            list_ = dico_D[j]

                            if(color in  list_):


                                list_.remove(color)


                                dico_D[j] = list_



    impossible_to_fill = 0

    for i in range(n):
        if (i not in list_removed_node):
            if(len( dico_D[i]) == 0):


                list_affected_colors.append(-1)
                list_removed_node.append(i)
                impossible_to_fill += 1

    print("nb cells impossible to fill")
    print(impossible_to_fill)


    nb_nodes_one_color = 0

    for i in range(n):
        if (i not in list_removed_node):
            if(len( dico_D[i]) == 1):
                nb_nodes_one_color += 1

    print("nb nodes one color")
    print(nb_nodes_one_color)


    original_node_list = []
    new_node_list = []

    cpt = 0

    nb_nodes = n - len(list_removed_node)

    sauvegarde_link = np.ones((nb_nodes,maxCol),dtype = np.int)*(-1)



    for i in range(n):

        if(i not in list_removed_node):

            original_node_list.append(i)

            new_node_list.append(cpt)

            for j in range(len(dico_D[i] )):

                sauvegarde_link[cpt,j] = int(dico_D[i][j])


            cpt += 1



    nb_max_col = -1

    for a in range(nb_nodes):

        cpt = 0
        for b in range(maxCol):
            if (sauvegarde_link[a, b] > -1):
                cpt += 1

        if(cpt > nb_max_col):
            nb_max_col = cpt


    sauvegarde_link = sauvegarde_link[:,:nb_max_col]


    # np.savetxt("preprocessed_LS/" + nameGraph + "_admissible_colors.csv", sauvegarde_link, delimiter=",")





    # mask = np.zeros((nb_nodes,nb_colors))
    #
    # for i in range(nb_nodes):
    #     for j in range(maxCol):
    #         if (sauvegarde_link[i, j] > -1):
    #             mask[i,sauvegarde_link[i, j]] = 1
    #
    # np.savetxt("benchmark_latin_square/" + nameGraph + "_mask_colors.csv", mask, delimiter=",")



    #
    # corresponding_colors = np.zeros((nb_colors,2))
    # for idx, val in enumerate(existing_colors):
    #     corresponding_colors[idx,0] = new_color[idx]
    #     corresponding_colors[idx,1] = val
    #


    #np.savetxt("test_solution_files/" + nameGraph + "_corresponding_colors.csv", corresponding_colors, delimiter=",")


    corresponding_nodes = np.zeros((nb_nodes,2), dtype = np.int)

    for idx, val in enumerate(original_node_list):

        corresponding_nodes[idx, 0] = int(new_node_list[idx])
        corresponding_nodes[idx, 1] = val


    #np.savetxt("test_solution_files/" + nameGraph + "_corresponding_nodes.csv", corresponding_nodes, delimiter=",")



    affected_colors = np.zeros((len(list_removed_node),2), dtype = np.int)
    for idx, val in enumerate(list_removed_node):

        affected_colors[idx, 0] = val
        affected_colors[idx, 1] = list_affected_colors[idx]


    #np.savetxt("test_solution_files/" + nameGraph + "_affected_colors.csv", affected_colors, delimiter=",")


    new_A = A[original_node_list,:][:,original_node_list]


    #np.savetxt("preprocessed_LS/" + nameGraph + "_new_adj.csv", new_A, delimiter=",")

    return new_A, sauvegarde_link, affected_colors, corresponding_nodes, impossible_to_fill




# Cuda kernel allowing to compute conflicts of D solutions in parallel
@cuda.jit
def computeUncoloredNodes( D,  tColor, nbConflicts):

    d = cuda.grid(1)

    if (d < D):
        f = 0
        for x in range(size):
            if(tColor[d,x] ==k or tColor[d,x] == -1):

                f += 1



        nbConflicts[d] = f



@cuda.jit
def removeRandomNodeInConflicts( rng_states, D, A, tColor):

    d = cuda.grid(1)

    if (d < D):

        tConflicts = nb.cuda.local.array((size), nb.int8)

        for x in range(size):
            tConflicts[x] = 0

        for x in range(size):
            for y in range(x):
                if(A[x,y]==1):

                    if ( tColor[d,x] == tColor[d,y] and tColor[d,x] < k):
                        tConflicts[x] += 1
                        tConflicts[y] += 1

        nbNodes_conflicts = 0

        for x in range(size):
            if(tConflicts[x] > 0 or tColor[d,x] == k):
                nbNodes_conflicts += 1

        nbToremove = int(nbNodes_conflicts*0.5)



        for x in range(size):
            if (tConflicts[x] > 0 or tColor[d,x] == k):

                if(nbToremove> 0):

                    if(nbNodes_conflicts == nbToremove):
                        tColor[d, x] = -1
                        nbToremove -= 1
                    else:
                        r = xoroshiro128p_uniform_float32(rng_states, d)

                        if(r < 0.5):
                            tColor[d, x] = -1
                            nbToremove -= 1

                    nbNodes_conflicts -= 1





@cuda.jit
def removeRandomNodeInConflicts( rng_states, D, A, tColor):

    d = cuda.grid(1)

    if (d < D):

        tConflicts = nb.cuda.local.array((size), nb.int8)

        for x in range(size):
            tConflicts[x] = 0

        for x in range(size):
            for y in range(x):
                if(A[x,y]==1):

                    if ( tColor[d,x] == tColor[d,y] and tColor[d,x] < k):
                        tConflicts[x] += 1
                        tConflicts[y] += 1

        nbNodes_conflicts = 0

        for x in range(size):
            if(tConflicts[x] > 0 or tColor[d,x] == k):
                nbNodes_conflicts += 1

        nbToremove = int(nbNodes_conflicts*0.5)



        for x in range(size):
            if (tConflicts[x] > 0 or tColor[d,x] == k):

                if(nbToremove> 0):

                    if(nbNodes_conflicts == nbToremove):
                        tColor[d, x] = -1
                        nbToremove -= 1
                    else:
                        r = xoroshiro128p_uniform_float32(rng_states, d)

                        if(r < 0.5):
                            tColor[d, x] = -1
                            nbToremove -= 1

                    nbNodes_conflicts -= 1


@cuda.jit
def fileNodeNoColorv2( rng_states, D, tColor):

    d = cuda.grid(1)

    if (d < D):

        for i in range(size):
            if (tColor[d,i] < 0):

                tColor[d,i] = k



@cuda.jit
def fileNodeNoColorGCP( rng_states, D, tColor):

    d = cuda.grid(1)

    if (d < D):

        for i in range(size):
            if (tColor[d,i] < 0):

                r = int(k * xoroshiro128p_uniform_float32(rng_states, d))

                tColor[d,i] = r



@cuda.jit
def fileNodeNoColorLS( rng_states, D, tColor, L, V):

    d = cuda.grid(1)

    if (d < D):

        for i in range(size):
            if (tColor[d,i] == k):

                nb_col = int(V[i])

                r = int(nb_col * xoroshiro128p_uniform_float32(rng_states, d))

                tColor[d,i] = int(L[i, r])




@cuda.jit
def resizeColors( rng_states, D, tColor, nbColorsInit):

    d = cuda.grid(1)

    if (d < D):

        for i in range(size):
            if (tColor[d,i] >= nbColorsInit):
                r = int(nbColorsInit * xoroshiro128p_uniform_float32(rng_states, d))
                if (r >= nbColorsInit):
                    r = nbColorsInit - 1

                tColor[d,i] = r



@cuda.jit
def removeconflictsMSCP( D, A, tColor, current_k):

    d = cuda.grid(1)

    if (d < D):

        tConflicts = nb.cuda.local.array((size), nb.int8)
        for i in range(size):
            tConflicts[i] = 0

        for i in range(size):
            for j in range(i,size):
                if(A[i,j] == 1):
                    if(tColor[d,i] == tColor[d,j]):
                        tConflicts[i] += 1
                        tConflicts[j] += 1

        vInterditsNewAddedClass = nb.cuda.local.array((size, k), nb.int8)
        for i in range(size):
            for j in range(k):
                vInterditsNewAddedClass[i,j] = 0

        vInterditsNewAddedClassSize = 0

        for i in range(size):
            if(tConflicts[i] > 0):
                found = 0
                foundClassId = 0

                while(found == 0 and foundClassId < vInterditsNewAddedClassSize):

                    if (vInterditsNewAddedClass[i, foundClassId] == 0):
                        found = 1

                    foundClassId += 1


                classId = found * (foundClassId-1) + (1-found)*vInterditsNewAddedClassSize
                vInterditsNewAddedClassSize = found * vInterditsNewAddedClassSize + (1-found)*(vInterditsNewAddedClassSize+1)

                for j in range(size):
                    if(A[i,j] == 1):
                        vInterditsNewAddedClass[j, classId] = 1

                        if(tColor[d,i] == tColor[d,j]):
                            tConflicts[j] -= 1

                tColor[d, i] = current_k + classId

        #nbColors[d] = nbColors[d]  + vInterditsNewAddedClassSize





@cuda.jit
def sortColors( rng_states, D, A, tColor, sumColorations):

    d = cuda.grid(1)

    if (d < D):


        tSizeOfColors = nb.cuda.local.array(size, nb.int8)
        tColor_sorted = nb.cuda.local.array(size, nb.int8)

        for j in range(k):
            tSizeOfColors[j] = 0

        for i in range(size):
            tColor_sorted[i] = -1


        for i in range(size):
            tSizeOfColors[int(tColor[d, i])] += 1

        for j in range(k):

            valMax = -1
            colorMax = -1

            startColor = int(k * xoroshiro128p_uniform_float32(rng_states, d))

            for l in range(k):
                color = (startColor + l) % k
                currentVal = tSizeOfColors[color]

                if (currentVal > valMax):
                    valMax = currentVal
                    colorMax = color

            for i in range(size):
                if (tColor[d, i] == colorMax and tColor_sorted[i] < 0):
                    tColor_sorted[i] = j

                    tSizeOfColors[int(tColor[d, i])] -= 1

        sumColor = 0
        for i in range(size):
            tColor[d, i] = tColor_sorted[i]
            sumColor += tColor[d, i] + 1


        sumColorations[d] = sumColor

# Cuda kernel allowing to compute conflicts of D solutions in parallel
@cuda.jit
def computeConflicts( D, A, tColor, nbConflicts):

    d = cuda.grid(1)

    if (d < D):
        f = 0
        for x in range(size):
            for y in range(x):
                if(A[x,y]==1):

                    if (tColor[d,x] > -1 and tColor[d,x] < k and tColor[d,x] == tColor[d,y]):
                        f += 1


        nbConflicts[d] = f



# Cuda kernel allowing to compute conflicts of D solutions in parallel
@cuda.jit
def computeSpins( D, tColor, Spin):

    d = cuda.grid(1)

    if (d < D):

        for x in range(size):
            for y in range(size):

                if (tColor[d,x] == tColor[d,y]):
                    Spin[d,x,y] = -1
                else:
                    Spin[d, x, y] =1


# Cuda kernel allowing to compute conflicts of D solutions in parallel
@cuda.jit
def computeV( D, tColor, V):

    d = cuda.grid(1)

    if (d < D):

        for x in range(size):
            for y in range(size):

                if (tColor[d,x] == tColor[d,y]):
                    V[d,x,y] = 1
                else:
                    V[d, x, y] = 0


# Cuda kernel allowing to compute conflicts of D solutions in parallel
@cuda.jit
def computeConflictsV2( D, A, tColor, nbConflicts):

    d = cuda.grid(1)

    if (d < D):
        f = 0
        for x in range(size):
            for y in range(x):
                if(A[x,y]==1):

                    if (tColor[d,x] == tColor[d,y]):
                        f += 1


        nbConflicts[d] = f

# Cuda kernel allowing to remove conflicts of D solutions in parallel
@cuda.jit
def removeConflictsv2(rng_states, D, A, tColor):

    d = cuda.grid(1)

    if (d < D):

        startNode = int(size * xoroshiro128p_uniform_float32(rng_states, d));

        for i in range(size):
            x = (startNode + i) % size
            if(tColor[d,x] < k):
                for y in range(size):
                    if(A[x,y]==1 and tColor[d,x] == tColor[d,y]):
                        tColor[d, x] = k



# Cuda kernel allowing to remove conflicts of D solutions in parallel
@cuda.jit
def removeConflicts(rng_states, D, A, tColor):

    d = cuda.grid(1)

    if (d < D):

        startNode = int(size * xoroshiro128p_uniform_float32(rng_states, d));

        for i in range(size):
            x = (startNode + i) % size
            if(tColor[d,x] < k):
                for y in range(x):
                    if(A[x,y]==1 and tColor[d,x] == tColor[d,y]):
                        tColor[d, y] = -1


# Cuda kernel allowing to remove conflicts of D solutions in parallel
@cuda.jit
def removeTooBigGroups(rng_states, D, maxCol, tColor):

    d = cuda.grid(1)

    if (d < D):

        startNode = int(size * xoroshiro128p_uniform_float32(rng_states, d));


        tSizeOfColors = nb.cuda.local.array(k, nb.int16)

        for j in range(k):
            tSizeOfColors[j] = 0


        for i in range(size):
            x = (startNode + i) % size
            if(tColor[d,x] > -1):
                tSizeOfColors[int(tColor[d, x])] += 1

                if(tSizeOfColors[int(tColor[d, x])] > maxCol):
                    tColor[d, x] = -1




@cuda.jit  
def getNeighbors(size_pop, nb_neighbors, matrixDistance, matrix_neighbors):

    d = cuda.grid(1)

    if (d < size_pop):
    
        for i in range(nb_neighbors):
            closest = -1
            min_dist = 999
            
            for j in range(size_pop):
                dist = matrixDistance[d,j]
                if(dist < min_dist):
                    min_dist = dist
                    closest = j
                    
            matrix_neighbors[d,i] = closest
            matrixDistance[d,closest] = size*2
            
            
            

# Create two random set of solution with color permutations (Used only to test NN invariance by color permutation)
def createRandomBatchPermutationSate(b, n,k):

    state = np.zeros((b,n,k))
    state2 = np.zeros((b, n, k))

    for i in range(b):
        for v in range(n):
            c = np.random.randint(k)
            state[i,v,c] = 1

            state2[i, v, k-c-1] = 1


    return state, state2
