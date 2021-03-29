from __future__ import print_function, absolute_import
from numba import cuda
import numpy
import math
import numba as nb
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import datetime
from random import shuffle
import random
import argparse

import logging
from time import time
from tqdm import tqdm

import crossovers.crossovers_numba
import distance_tools.distance_numba
import local_searches.tabuCol_numba
import utils.tools_numba

from distance_tools.MACol_insertion_algorithm import insertion_pop
from joblib import Parallel, delayed



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('id_gpu', metavar='t', type=str, help='id_gpu')
    parser.add_argument('id_graph', metavar='t', type=str, help='id_graph')
    parser.add_argument('k', metavar='t', type=str, help='k')
    parser.add_argument('r', metavar='t', type=str, help='r')
    parser.add_argument('type_instance', metavar='t', type=str, help='type_instance')
    parser.add_argument('seed', metavar='t', type=str, help='seed')

    args = parser.parse_args()

    budget_time_total = 3600*12

    ######## Init gpu devices
    id_gpu = int(args.id_gpu)
    nb.cuda.select_device(id_gpu)
    device = "cuda:" + str(id_gpu)
    print(device)

    

    ######## Load graph
    type_instance = args.type_instance

    if(type_instance == "PLSE"):

        k = int(args.k)
        r = int(args.r)

        nameGraph = "QC-" + str(k) + "-"+ str(r) +"-"+ str(args.id_graph) + ".txt"
        filepath = "benchmark/PLSE_" + str(k) + "_"+ str(r) +"_1/"
        kernalization = False

    elif(type_instance == "QWH"):

        k = int(args.k)
        r = int(args.r)

        nameGraph = "QWH-" + str(k) + "-"+ str(r) +"-"+ str(args.id_graph) + ".txt"
        filepath = "benchmark/LSC_" + str(k) + "_"+ str(r) + "/"
        kernalization = False

    elif(type_instance == "Tradi"):

        filepath = "benchmark/Traditional_instances/"
          
        if(int(args.id_graph) == 1):
            nameGraph = "qwhdec.order35.holes405.1.col"
            k = 35
        if(int(args.id_graph) == 2):
            nameGraph = "qwhdec.order33.holes381.bal.1.col"
            k = 33
        if(int(args.id_graph) == 3):
            nameGraph = "qwhdec.order30.holes320.1.col"
            k = 30
        if(int(args.id_graph) == 4):
            nameGraph = "qwhdec.order30.holes316.1.col"
            k = 30
        if(int(args.id_graph) == 5):
            nameGraph = "qwhdec.order70.holes2940.1.col"
            k = 70    
        if(int(args.id_graph) == 6 ):
            nameGraph = "qwhdec.order70.holes2450.1.col"
            k = 70
            normalSizeGraph = False
        if(int(args.id_graph) == 7 ):
            nameGraph = "qwhdec.order60.holes1620.1.col"
            k = 60
        if(int(args.id_graph) == 8 ):
            nameGraph = "qwhdec.order60.holes1440.1.col"
            k = 60
        if(int(args.id_graph) == 9 ):
            nameGraph = "qwhdec.order60.holes1152.bal.1.col"
            k = 60
        if(int(args.id_graph) == 10 ):
            nameGraph = "qwhdec.order60.holes1080.bal.1.col"
            k = 60
        if(int(args.id_graph) == 11 ):
            nameGraph = "qwhdec.order50.holes825.bal.1.col"
            k = 50
        if(int(args.id_graph) == 12 ):
            nameGraph = "qwhdec.order50.holes750.bal.1.col"
            k = 50
        if(int(args.id_graph) == 13 ):
            nameGraph = "qwhdec.order40.holes528.1.col"
            k = 40
        if(int(args.id_graph) == 14 ):
            nameGraph = "qwhdec.order5.holes10.1.col"
            k = 5
        if(int(args.id_graph) == 15 ):
            nameGraph = "qwhdec.order18.holes120.1.col"
            k = 18
        if(int(args.id_graph) == 16 ):
            nameGraph = "qg.order30.col"
            k = 30
        if(int(args.id_graph) == 17 ):
            nameGraph = "qg.order40.col"
            k = 40
        if(int(args.id_graph) == 18 ):
            nameGraph = "qg.order60.col"
            k = 60
        if(int(args.id_graph) == 19 ):
            nameGraph = "qg.order100.col"
            k = 100
                                                         
        kernalization = False


    A, listCols, affected_colors, corresponding_nodes, nb_impossible_to_fill = utils.tools_numba.preprocess_PLSE_instance(filepath, nameGraph, k, kernalization)

    size = A.shape[0]

    bigSizeGraph = False

    if(size > 2000):
        bigSizeGraph = True



    listCols_global_mem = cuda.to_device(np.ascontiguousarray(listCols))
    vect_nb_admissible_colors = np.zeros((size))

    for i in range(size):
        cpt = 0
        for j in range(int(listCols.shape[1])):
            if(listCols[i,j] != -1):
                cpt+= 1
        vect_nb_admissible_colors[i] = cpt

    vect_nb_admissible_colors_mem = cuda.to_device(vect_nb_admissible_colors)

    A_global_mem = cuda.to_device(A) # load adjacency matrix to device

    print("size : " + str(size))
    
    
    ######## Parameters
    min_dist_insertion = size/10
    max_iter_ITS = 10
    nb_iter_tabu = int(size*100/max_iter_ITS) # Number of local search iteration with TabuCol algorithm

    size_pop = 4096*3 # Size of the memetic population
    

    
    alpha = 0.6 
    nb_neighbors = 1
    ############


    modeTest = False

    if(modeTest):
        print("TEST")
        nb_iter_tabu = 10
        size_pop = 10
        nb_neighbors = 3

    print(nb_iter_tabu)

    # Numba parameters
    threadsperblock = 64

    best_score = 9999

    ######### Init logs
    date = datetime.datetime.now()

    if (modeTest):
        name_expe = "test"
    else:
        name_expe = "LCS_" + "_size_pop_" + str(size_pop)  + "_nb_iter_" + str(nb_iter_tabu) + "_nb_neighbors_" + str(nb_neighbors)  + "_" + nameGraph + "_k_" + str(k) + "_" + str(date) + ".txt"

    logging.basicConfig(filename= "logs/" + name_expe + ".log",level=logging.INFO)
    #########

    ### Init tables ######

    offsprings_pop = np.zeros((size_pop, size), dtype=np.int32) # new colors generated after offspring

    fitness_pop = np.ones((size_pop), dtype=np.int32)*9999 # vector of fitness of the population
    fitness_offsprings = np.zeros((size_pop),dtype=np.int32) # vector of fitness of the offsprings

    matrice_crossovers_already_tested = np.zeros((size_pop, size_pop), dtype=np.uint8)
    matrice_crossovers_already_tested = np.zeros((size_pop, size_pop), dtype=np.uint8)

    tabuTenure_tabucol = np.zeros((size_pop, size, k), dtype=np.int32)

    if(bigSizeGraph):
        gamma_tabucol = np.zeros((size_pop, size, k), dtype=np.int16)
        gamma_tabucol_gpu_memory = cuda.to_device(gamma_tabucol)

    # Big Distance matrix with all individuals in pop and all offsprings at each generation
    matrixDistanceAll = np.zeros(( 2 * size_pop, 2 * size_pop), dtype=np.int16)

    matrixDistanceAll[:size_pop, :size_pop] = np.ones(( size_pop, size_pop), dtype=np.int16)*9999

    # Consider renaming these matrices
    matrixDistance1 = np.zeros((size_pop, size_pop), dtype=np.int16) # Matrix with ditances between individuals in pop and offsprings
    matrixDistance2 = np.zeros((size_pop, size_pop), dtype=np.int16) # Matrix with ditances between all offsprings

    fit_crossover = np.zeros((size_pop), dtype=np.float32)


    ## Send matrices to GPU device
    
    offsprings_pop_gpu_memory = cuda.to_device(offsprings_pop)
    fitness_pop_gpu_memory = cuda.to_device(fitness_pop)
    fitness_offsprings_gpu_memory = cuda.to_device(fitness_offsprings)
    tabuTenure_gpu_memory = cuda.to_device(tabuTenure_tabucol)

    matrixDistance1_gpu_memory = cuda.to_device(matrixDistance1)
    matrixDistance2_gpu_memory = cuda.to_device(matrixDistance2)

    fit_crossover_global_mem = cuda.to_device(fit_crossover)

    crossovers.crossovers_numba.size = size
    crossovers.crossovers_numba.k = k
    distance_tools.distance_numba.size = size
    distance_tools.distance_numba.k = k
    distance_tools.distance_numba.kplus1 = k +1
    local_searches.tabuCol_numba.size = size
    local_searches.tabuCol_numba.k = k
    utils.tools_numba.size = size
    utils.tools_numba.k = k
    utils.tools_numba.size_pop = size_pop
     
    #################################


    ##### Configure the different sizes of blocks for numba
    blockspergrid1 = (size_pop + (threadsperblock - 1)) // threadsperblock
    blockspergrid2 = (size_pop*size_pop + (threadsperblock - 1)) // threadsperblock
    blockspergrid3 = (size_pop * (size_pop - 1) // 2 + (threadsperblock - 1)) // threadsperblock

    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid1, seed=int(args.seed))  ## Init numba random generator


    time_start = time()


    ### Init pop ####

    utils.tools_numba.initPop[blockspergrid1, threadsperblock](rng_states, size_pop, offsprings_pop_gpu_memory, listCols_global_mem, vect_nb_admissible_colors_mem)
    colors_pop = offsprings_pop_gpu_memory.copy_to_host()
    ######


    for epoch in range(100000):

        ####### First step : local search  ################################################################################
        ########################################################################################################

        #### Start tabu

        logging.info("############################")
        logging.info("Start ITS")
        logging.info("############################")

        startEpoch = time()

        print("start ITS")

        start = time()

        offsprings_pop_after_tabu = np.zeros((size_pop, size), dtype=np.int32)
        fitness_offsprings_after_tabu  = np.ones((size_pop), dtype=np.int32)*9999


        for iter in range(max_iter_ITS):

            print("start tabu")

 
            if(bigSizeGraph):
                local_searches.tabuCol_numba.tabucol_LatinSquare_BigSize[blockspergrid1, threadsperblock](rng_states, size_pop, nb_iter_tabu, A_global_mem, listCols_global_mem, vect_nb_admissible_colors_mem,
                                                        offsprings_pop_gpu_memory, fitness_offsprings_gpu_memory, tabuTenure_gpu_memory, alpha, gamma_tabucol_gpu_memory)    
            else:
                local_searches.tabuCol_numba.tabucol_LatinSquare[blockspergrid1, threadsperblock](rng_states, size_pop, nb_iter_tabu, A_global_mem, listCols_global_mem, vect_nb_admissible_colors_mem,
                                                        offsprings_pop_gpu_memory, fitness_offsprings_gpu_memory, tabuTenure_gpu_memory, alpha)


            nb.cuda.synchronize()

            logging.info("Tabucol duration : " + str(time() - start))

            offsprings_pop = offsprings_pop_gpu_memory.copy_to_host()
            fitness_offsprings = fitness_offsprings_gpu_memory.copy_to_host()

            best_score_pop = np.min(fitness_offsprings)
            worst_score_pop = np.max(fitness_offsprings)
            avg_pop = np.mean(fitness_offsprings)

            logging.info("Epoch : " + str(epoch) + " iter : " + str(iter))
            logging.info("Pop best : " + str(best_score_pop) + "_worst : " + str(worst_score_pop) + "_avg : " + str(avg_pop))


            for i in range(size_pop):
                if(fitness_offsprings[i] < fitness_offsprings_after_tabu[i]):
                    fitness_offsprings_after_tabu[i] = fitness_offsprings[i]
                    offsprings_pop_after_tabu[i,:] = offsprings_pop[i,:]

            if(min(fitness_offsprings_after_tabu) < 1 or (nb_impossible_to_fill == 1 and min(fitness_offsprings_after_tabu) < 2)):
                break

            print("end tabu")


            if(iter < max_iter_ITS - 1):

                utils.tools_numba.removeRandomNodeInConflicts[blockspergrid1, threadsperblock](rng_states, size_pop, A_global_mem,
                                                                                               offsprings_pop_gpu_memory)

                if(bigSizeGraph):
                    local_searches.tabuCol_numba.tabucol_LatinSquareNotAffectedNode_BigSize[blockspergrid1, threadsperblock](rng_states, size_pop, nb_iter_tabu, A_global_mem, listCols_global_mem, vect_nb_admissible_colors_mem,
                                                        offsprings_pop_gpu_memory, fitness_offsprings_gpu_memory, tabuTenure_gpu_memory, alpha, gamma_tabucol_gpu_memory) 

                else:
                    local_searches.tabuCol_numba.tabucol_LatinSquareNotAffectedNode[blockspergrid1, threadsperblock](rng_states, size_pop, nb_iter_tabu, A_global_mem, listCols_global_mem, vect_nb_admissible_colors_mem,
                                                        offsprings_pop_gpu_memory, fitness_offsprings_gpu_memory, tabuTenure_gpu_memory, alpha)

                utils.tools_numba.fileNodeNoColor[blockspergrid1, threadsperblock](rng_states, size_pop, offsprings_pop_gpu_memory, listCols_global_mem, vect_nb_admissible_colors_mem)



   
        nb.cuda.synchronize()

        print("end tabu")
        logging.info("Tabucol duration : " + str(time() - start))

        offsprings_pop = offsprings_pop_gpu_memory.copy_to_host()
        fitness_offsprings = fitness_offsprings_gpu_memory.copy_to_host()
       
        logging.info("Tabucol duration : " + str(time() - start))
        #######################################################


        ######### Get and log results

        logging.info("############################")
        logging.info("Results TabuCol")
        logging.info("############################")


        #### Remove conflicts

        offsprings_pop_gpu_memory = cuda.to_device(offsprings_pop_after_tabu)
        utils.tools_numba.greedyRemoveConflicts[blockspergrid1, threadsperblock](rng_states, size_pop, A_global_mem, offsprings_pop_gpu_memory)
    
        utils.tools_numba.computeUncoloredNodes[blockspergrid1, threadsperblock]( size_pop, offsprings_pop_gpu_memory, fitness_offsprings_gpu_memory)
        fitness_offsprings_after_tabu = fitness_offsprings_gpu_memory.copy_to_host()

        print("fitness_offsprings_after_tabu")
        print(fitness_offsprings_after_tabu)


        offsprings_pop_after_tabu = offsprings_pop_gpu_memory.copy_to_host()

        min_plse_score = np.min(fitness_offsprings_after_tabu)

        print("min_plse_score")
        print(min_plse_score + nb_impossible_to_fill )

        logging.info("min_plse_score")
        logging.info(min_plse_score + nb_impossible_to_fill)

        if (min_plse_score < best_score):

            best_score = min_plse_score

            logging.info("Save best solution")

            GCP_solution = offsprings_pop_after_tabu[np.argmin(fitness_offsprings_after_tabu)]


        if (epoch % 1 == 0):

            fichier = open("Evol/" + name_expe, "a")

            fichier.write("\n" + str(best_score + nb_impossible_to_fill) + "," + str(
                    min_plse_score + nb_impossible_to_fill) + "," + str(epoch))

            fichier.close()

        if(time() - time_start >budget_time_total or best_score < 1 or (best_score < 2 and nb_impossible_to_fill == 1)):
            break



        ####### Second step : insertion of offsprings in pop according to diversity/fit criterion ###############
        ########################################################################################################

        logging.info("Keep best with diversity/fit tradeoff")

        print("start matrix distance")
        logging.info("start matrix distance")

        start = time()


        colors_pop_gpu_memory = cuda.to_device(colors_pop)


        blockspergrid_test = ((size_pop * size_pop) + (threadsperblock - 1)) // threadsperblock



        distance_tools.distance_numba.computeMatrixDistance_PorumbelApprox[blockspergrid_test, threadsperblock](size_pop,
                                                                                                     size_pop,
                                                                                                     matrixDistance1_gpu_memory,
                                                                                                     colors_pop_gpu_memory,
                                                                                                     offsprings_pop_gpu_memory)

        matrixDistance1 = matrixDistance1_gpu_memory.copy_to_host()

        blockspergrid_test = ((size_pop * size_pop) + (threadsperblock - 1)) // threadsperblock

        distance_tools.distance_numba.computeSymmetricMatrixDistance_PorumbelApprox[blockspergrid_test, threadsperblock](size_pop, matrixDistance2_gpu_memory, offsprings_pop_gpu_memory)

        matrixDistance2 = matrixDistance2_gpu_memory.copy_to_host()

        # Aggregate all the matrix in order to obtain a full 2*size_pop matrix with all the distances between individuals in pop and in offspring
        matrixDistanceAll[:size_pop, size_pop:] = matrixDistance1
        matrixDistanceAll[size_pop:, :size_pop] = matrixDistance1.transpose(1, 0)
        matrixDistanceAll[size_pop:, size_pop:] = matrixDistance2


        logging.info("Matrix distance duration : " + str(time() - start))

        print("end  matrix distance")
        #####################################


        print("start insertion in pop")
        start = time()


        results = insertion_pop(size_pop,matrixDistanceAll, colors_pop, offsprings_pop_after_tabu, fitness_pop,
                                         fitness_offsprings_after_tabu,   matrice_crossovers_already_tested,  min_dist_insertion )


        matrixDistanceAll[:size_pop,: size_pop] = results[0]
        fitness_pop = results[1]
        colors_pop = results[2]
        matrice_crossovers_already_tested = results[3]


        logging.info("Insertion in pop : " + str(time() - start))

        print("end insertion in pop")


        logging.info("After keep best info")


        best_score_pop = np.min(fitness_pop)
        worst_score_pop = np.max(fitness_pop)
        avg_score_pop = np.mean(fitness_pop)

        logging.info("Pop _best : " + str(best_score_pop) + "_worst : " + str(worst_score_pop) + "_avg : " + str(avg_score_pop))
        logging.info(fitness_pop)

        matrix_distance_pop = matrixDistanceAll[:size_pop, :size_pop]

        max_dist = np.max(matrix_distance_pop)
        min_dist = np.min(matrix_distance_pop + np.eye(size_pop) * 9999)
        avg_dist = np.sum(matrix_distance_pop) / (size_pop * (size_pop - 1))

        logging.info("Avg dist : " + str(avg_dist) + " min dist : " + str(min_dist) + " max dist : " + str(max_dist))


        ####### Third step : selection of best crossovers to generate new offsprings  #########################
        ########################################################################################################

        logging.info("############################")
        logging.info("start crossover")
        logging.info("############################")

        print("start crossover")

        start = time()
        
        bestColor_global_mem = cuda.to_device(colors_pop)



        best_cross = np.where(matrice_crossovers_already_tested == 1, 9999,
                              matrixDistanceAll[:size_pop, :size_pop])
        best_cross = np.where(best_cross == 0, 9999, best_cross)


        closest_individuals = np.argsort(best_cross, axis=1)[ :, :1]
        closest_individuals_gpu_memory = cuda.to_device(np.ascontiguousarray(closest_individuals))


        matrice_crossovers_already_tested_gpu_memory = cuda.to_device(matrice_crossovers_already_tested)




        crossovers.crossovers_numba.compute_nearest_neighbor_crossovers_Latin_Square[blockspergrid1, threadsperblock](
                         rng_states,
                size_pop,
                vect_nb_admissible_colors_mem,
                listCols_global_mem,
                bestColor_global_mem,
                offsprings_pop_gpu_memory,
                matrice_crossovers_already_tested_gpu_memory,
                closest_individuals_gpu_memory)


        matrice_crossovers_already_tested = matrice_crossovers_already_tested_gpu_memory.copy_to_host()


        logging.info("nb cross already tested in pop : " + str(np.sum(matrice_crossovers_already_tested)))
            
        logging.info("Crossover duration : " + str(time() - start))

        print("end crossover")

        logging.info("generation duration : " + str(time() - startEpoch ))
        print("generation duration : " + str(time() - startEpoch ))


    ################################################################

    ### Post processing best solution Latin square
        
    print("store best result in original format")
    nb_cells = k*k

    LS_solution = np.ones((nb_cells,)) * (-1)

    for i in range(nb_cells):
        color = -1
        for j in range(affected_colors.shape[0]):
            if (affected_colors[j, 0] == i):
                color = affected_colors[j, 1]
        LS_solution[i] = color

    for i in range(GCP_solution.shape[0]):

        new_node_num = int(corresponding_nodes[i, 1])

        if (GCP_solution[i] < k):

            LS_solution[new_node_num] = int(GCP_solution[i])


    nb_empty_cells = 0
    for i in range(LS_solution.shape[0]):
        if(LS_solution[i] < 0):
            nb_empty_cells += 1

    print("Nb empty cells : " + str(nb_empty_cells))


    # Verify nb conflicts in Latin square
    A = np.zeros((nb_cells, nb_cells), dtype = int)

    for i in range(nb_cells):

        rowi = i%k
        coli = i//k

        for j in range(nb_cells):
            rowj = j % k
            colj = j // k

            if(((rowj == rowi) or  (coli == colj)) and i!=j):

                A[i,j] = 1


    nbConflicts = 0

    for v in range(A.shape[0]):

        colorV = LS_solution[v]

        if (colorV > -1):

            for i in range(A.shape[1]):

                if (A[v, i] == 1):

                    colorVprim = LS_solution[i]

                    if (colorVprim == colorV):
                        nbConflicts += 1

    print("Nb conflicts : " + str(nbConflicts))

    latin_square = np.zeros((k,k), dtype = int)

    for i in range(LS_solution.shape[0]):

        row = int(i % k)
        col = int(i // k)

        latin_square[row, col] = int(LS_solution[i])

    if(nbConflicts == 0):
        np.savetxt("legal_solutions_LS/legal_solution_latin_square_" + nameGraph + "_nb_empty_cells_" + str(nb_empty_cells) + ".csv", latin_square)







