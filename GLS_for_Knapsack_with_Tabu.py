﻿# Required Librarie
import math
import pandas as pd
import time 
import numpy as np
import random
import copy
from matplotlib import pyplot as plt 
from memory_profiler import profile
from random import Random, random
from math import exp

def Initial_solution(initial_prop_of_items):  #initial prop of items is proportion of items in initial solution #OK
    np.random.seed(1)    #seed
    x = np.random.binomial(1, initial_prop_of_items, size=n) # create nikoubunpu 
    return x  

def evaluate(x):#OK
    a = np.array(x)
    b = np.array(value)
    c = np.array(weights)

    totalValue = np.dot(a, b)  # compute the value of the knapsack selection
    totalWeight = np.dot(a, c)  # compute the weight value of the knapsack selection

    if totalWeight > maxWeight:
        return [totalWeight-maxWeight, totalWeight]
    else:
        return [totalValue, totalWeight]  # returns a list of both total value and total weight

def OneflipNeighborhood(x):#OK
    nbrhood = []

    for i in range(0, n):
        temp=list(x)
        nbrhood.append(temp)
        if nbrhood[i][i] == 1:
            nbrhood[i][i] = 0
        else:
            nbrhood[i][i] = 1

    return nbrhood

def taboo_search(max_super_best_steps):

    taboo_tenure = 30
    solutionsChecked = 0
    
    local_best_solution = copy.deepcopy(solution)
    f_super_best =  copy.deepcopy(solution)       #Best solution so far

    taboo_list=[0]*n            #taboo status of each element in solution

    count=0                     #counting number of non improving steps


    while (count< max_super_best_steps):

        Neighborhood = OneflipNeighborhood(solution[2])  # create a list of all neighbors in the neighborhood of x_curr
        neighbor=0      #Number of element changed in current step
        local_best_solution[0]=0     #Reseting best neighbour value to zero
        for s in Neighborhood:  # evaluate every member in the neighborhood of x_curr
            solutionsChecked=solutionsChecked+1

            #print evaluate(s)
            #print 'mememememememme'

            if (evaluate(s)[0] > local_best_solution[0]) and (taboo_list[neighbor]==0):  # and (evaluate(s)[1]< maxWeight):
                solution[2] = s[:]  # find the best member and keep track of that solution
                local_best_solution[0] = evaluate(s)[0]    #Best solution in neighbourhood
                local_best_solution[1] = evaluate(s)[1]
                neighbor_selected = neighbor   #neighbour selected in current step

            if (evaluate(s)[0]> f_super_best[0]):    #Updating best solution fourd so far
                solution[2] = s[:]
                local_best_solution[0] = evaluate(s)[0]
                local_best_solution[1] = evaluate(s)[1]

                f_super_best[0] = evaluate(s)[0]
                f_super_best[1] = evaluate(s)[1]

                x_super= s[:]
                neighbor_selected = neighbor
                change=1

            neighbor = neighbor + 1

        count = count + 1           #Counting number of steps with our improvement

        if(change == 1):            #Recording change status
            count=0
            change=0

        for i in range(0,len(taboo_list)-1):   #Updating taboo status of each item
            xx=taboo_list[i]
            if(xx>0):
                taboo_list[i]=xx-1

        taboo_list[neighbor_selected]=taboo_tenure   #Updating taboo status of selected item

    local_best_solution[2] = x_super

    return local_best_solution

#Function: Augmented Cost: f(x)
def augumented_cost(x_curr, penalty, limit):#OK
    augmented = 0   
    for i in range(0, len(penalty) - 1):
        c1 = value[i]
        c2 = weights[i]      
        
        if x_curr[i] == 1:
          augmented = augmented + (c2/c1) + (limit * penalty[i]) # kaiwohyouka   

    return augmented

# Function: Local Search #only calicurate the cost and obtain local-opt(value and weight and x)
def local_search(penalty, max_attempts, limit):

    x_curr = Initial_solution(0.1)  
    f_curr = evaluate(x_curr)  # f_curr will hold the evaluation of the current soluton
    f_best = f_curr[:]          #Best solution in neighbourhood

    f_super_best=f_curr[:]     #Best solution so far

    
    ag_cost = augumented_cost(x_curr, penalty = penalty, limit = limit) #initial

    opt_solution = copy.deepcopy(solution) #best solution
    candidate = copy.deepcopy(solution) 

    candidate = taboo_search(solution, max_attempts) # tabootenure 30
    candidate_augmented = augumented_cost(candidate[2], penalty = penalty, limit = limit)
       
    if candidate_augmented < ag_cost:
       opt_solution  = copy.deepcopy(candidate)
       ag_cost = augumented_cost(opt_solution[2], penalty = penalty, limit = limit)
                         
    return opt_solution 

#Function: Utility
def utility (x_curr, penalty, limit = 1): #OK
    utilities = [0] * len(penalty)
   
    for i in range(0, len(penalty) - 1):
        c1 = value[i]
        c2 = weights[i]      
        
        if x_curr[i] == 1:          
          utilities[i] = (c2/c1) /(1 + penalty[i])

    return utilities

#Function: Update Penalty
def update_penalty(penalty, utilities): #OK
    max_utility = max(utilities)   
    for i in range(0, len(penalty) - 1):
       
        if (utilities[i] == max_utility):
            penalty[i] = penalty[i] + 1   

    return penalty
    
# Function: Guided Search #only obtain the local_opt(value and weight and x)
#@profile
def guided_search(alpha, local_search_optima, max_attempts, iterations):

    t1 = time.time() 
    count = 0
    limit = alpha * (local_search_optima / 20 )  
    penalty = [0] * n 
 
    solution= [0] * 3
    solution[2] = copy.deepcopy(penalty)

    print solution

    best_solution = copy.deepcopy(solution)
    
    while (count < iterations):
        solution = local_search(penalty = penalty, max_attempts = max_attempts, limit = limit)
        utilities = utility(solution[2], penalty = penalty, limit = limit) 
        #print utilities
        penalty = update_penalty(penalty = penalty, utilities = utilities)
        #print penalty

        t2 = time.time() #add
        if (solution[0] < best_solution[0]):
            best_solution = copy.deepcopy(solution) 
        count = count + 1
        elapsed_time = t2-t1 #add
        print("Iteration = ", count, " value ", best_solution[0]," weights ", best_solution[1] ," Time ", elapsed_time) 
    
#----------------------------------------------------------------------------------------------------------
seed = 5113
myPRNG = Random(seed)
n = 100
# to get a random number between 0 and 1, use this:             myPRNG.random()
# to get a random number between lwrBnd and upprBnd, use this:  myPRNG.uniform(lwrBnd,upprBnd)
# to get a random integer between lwrBnd and upprBnd, use this: myPRNG.randint(lwrBnd,upprBnd)

# create an "instance" for the knapsack problem
value = []
for i in range(0, n):
    value.append(myPRNG.uniform(10, 100)) # make some value from 10 between 100 in float

weights = []
for i in range(0, n):
    weights.append(myPRNG.uniform(5, 20)) # make some weight from 5 between 20 in float
    
maxWeight = 5 * n 

# monitor the number of solutions evaluated
solutionsChecked = 0

# Call the Function
guided_search(alpha = 0.5, local_search_optima = 100, max_attempts = 30, iterations = 10) #change iterationsnumber to 10 from 2500