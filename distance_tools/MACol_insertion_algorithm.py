
import numpy as np
from random import shuffle
import random




def insertion_pop(size_pop,  matrixDistanceAll, colors_pop,
                           offsprings_pop_after_tabu, fitness_pop, fitness_offsprings_after_tabu, matrice_crossovers_already_tested, min_dist):



    all_scores = np.hstack((fitness_pop, fitness_offsprings_after_tabu))

    matrice_crossovers_already_tested_new= np.zeros((size_pop*2,size_pop*2), dtype = np.uint8)
    matrice_crossovers_already_tested_new[:size_pop,:size_pop] = matrice_crossovers_already_tested

    idx_best = np.argsort(all_scores)

    idx_selected = []

    cpt = 0

    for i in range(0,size_pop * 2):

        idx = idx_best[i]

        if(len(idx_selected) > 0):
            dist = np.min(matrixDistanceAll[idx,idx_selected])
        else: 
            dist = 9999
         
        if (dist >= min_dist):


            idx_selected.append(idx)

            if(idx >= size_pop):
                cpt+=1

        if(len(idx_selected) == size_pop):
            break;

    print("len(idx_selected)")
    print(len(idx_selected))


    if(len(idx_selected) != size_pop):
        for i in range(0,size_pop * 2):
            idx = idx_best[i]

            if(idx not in idx_selected):
                dist = np.min(matrixDistanceAll[idx,idx_selected])
                if (dist >= 0):
                    idx_selected.append(idx)

            if(len(idx_selected) == size_pop):
                break;


    print("Nb insertion " + str(cpt))


    new_matrix = matrixDistanceAll[idx_selected, :][:,idx_selected]




    stack_all = np.vstack((colors_pop, offsprings_pop_after_tabu))


    colors_pop_v2 = stack_all[idx_selected]


    fitness_pop_v2 = all_scores[idx_selected]

    matrice_crossovers_already_tested_v2 = matrice_crossovers_already_tested_new[idx_selected, :][:, idx_selected]


    return new_matrix, fitness_pop_v2,   colors_pop_v2, matrice_crossovers_already_tested_v2



