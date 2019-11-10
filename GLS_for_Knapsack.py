import pandas as pd
import time # add
import numpy as np
import random
import copy
from matplotlib import pyplot as plt 
from memory_profiler import profile

# Function: Initial Solution: create rndom solution #kokokakuninn
def Random_select_function(X, item_id, item_len): 

    list_s = [0] * 2
    list_n = []
    
    while(list_s[1] < Max_weight):
         m = random.choice(item_id)    # select randomly one number(0 <= 99)
         list_n.append(m)                          
         list_s[0] = list_s[0] + X[m][0]        # calicurate total_value
         list_s[1] = list_s[1] + X[m][1] 
         # print list_s[1]       # calicurate total_weight
         item_id.remove(m)
         # print item_id
    
    list_s[0] = list_s[0] - X[m][0]     # koko   
    list_s[1] = list_s[1] - X[m][1] 
    list_n.remove(m)
    item_id.append(m)
    return list_s, list_n, item_id
    
# Function: search
def search(X, id, solution, item_n):
    
    flag = 0
    local_solution = copy.deepcopy(solution)
    local_solution_item = copy.deepcopy(item_n)
    # print local_solution_item
    n = random.choice(local_solution_item)
    local_solution_item.remove(n)
    id.append(n)
    # print len(id)
    # print len(local_solution_item), len(id)
    local_solution[0] = local_solution[0] - X[n][0]
    local_solution[1] = local_solution[1] - X[n][1]


    while(local_solution[1] < Max_weight):
         #print id  
         m = random.choice(id)
         local_solution_item.append(m)                          # select randomly one number(0 <= 99)
         # print local_solution_item
         local_solution[0] = local_solution[0] + X[m][0]        # calicurate total_value
         local_solution[1] = local_solution[1] + X[m][1]        # calicurate total_weight
         id.remove(m)
         flag = 1
    
    if flag == 1:
      local_solution[0] = local_solution[0] - X[m][0]        
      local_solution[1] = local_solution[1] - X[m][1] 
      local_solution_item.remove(m)
      id.append(m)

    # print local_solution

    return local_solution, local_solution_item, id                   

#Function: Augmented Cost: f(x)
def augumented_cost(X, item_n, penalty, limit):

    augmented = 0   

    for i in range(len(item_n)): 
        augmented = augmented + X[item_n[i]][0]  - (limit * penalty[item_n[i]])  

    return augmented

# Function: Local Search #
def local_search(X, id, solution, item_n, item_len, penalty, max_attempts, limit):
    count = 0
    ag_cost = 0

    local_solution = copy.deepcopy(solution)
    local_solution_item = copy.deepcopy(item_n)
    local_solution_id = copy.deepcopy(id)
    
    cabdidarte = copy.deepcopy(solution)

    while (count < max_attempts):
        candidate, candidate_item, candidate_id = search(X, local_solution_id, local_solution, local_solution_item)
        candidate_augmented = augumented_cost(X, candidate_item, penalty, limit)       
        if candidate_augmented > ag_cost:
            local_solution  = copy.deepcopy(candidate)
            local_solution_item = copy.deepcopy(candidate_item)
            local_solution_id = copy.deepcopy(candidate_id)
            ag_cost = candidate_augmented
            count = 0
        else:
            count = count + 1                             
    return local_solution, local_solution_item, local_solution_id

#Function: Utility
def utility (X, penalty, item_n): # caricurate utilities of number of item_n # maybe OK

    utilities = [0] * len(penalty)
   
    for i in range(len(item_n)):
        c1 = X[item_n[i]][0]
        c2 = X[item_n[i]][1]      
        utilities[item_n[i]] = (c2/c1) /(1 + penalty[item_n[i]])

    return utilities

#Function: Update Penalty
def update_penalty(penalty, utilities): #maybe OK

    max_utility = max(utilities)   
    for i in range(len(penalty)): 
       
       if (utilities[i] == max_utility):
            penalty[i] = penalty[i] + 1   

    return penalty
    
# Function: Guided Search
@profile
def guided_search(X, seed, id, item_n, item_len, max_attempts, iterations):

    t1 = time.time()
    count = 0
    limit = 2.5 
    penalty = [0 for i in range(item_len)] # make penalty for each items number list
    utilities = [0 for i in range(item_len)] # make penalty for each items utility list

    solution = copy.deepcopy(seed) # I shuould identify structure solution
    solution_item = copy.deepcopy(item_n) 
    solution_id = copy.deepcopy(id)

    best_solution = copy.deepcopy(seed) 
    best_solution_item = []

    while (count < iterations):
        # print count
        solution, solution_item, solution_id = local_search(X, solution_id, solution, solution_item, item_len ,penalty, max_attempts, limit)          # get local solution
        utilities = utility(X, penalty, solution_item)       # evaluate local
        penalty = update_penalty(penalty, utilities)
        t2 = time.time() #add

        if (solution[0] > best_solution[0]):           # if value > best value
            best_solution = copy.deepcopy(solution) 

        count = count + 1
        elapsed_time = t2-t1 #add
        print("Iteration = ", count, " Total value and weight ", best_solution, " Time ", elapsed_time) 

    return best_solution

#--------- main part

# Load File - A Distance Matrix: I should identify items structure
X = pd.read_csv('sample4.txt', sep = '\t')

X = X.values   # X has 0:value and 1:weight
# print np.shape(X)
Max_weight = 10000  # max content of knapsack
#print np.shape(X)[0]
item_len = np.shape(X)[0] # get full number of items
item_id = [i for i in range(item_len) ] 

# seed = [0] * 2 # seed is random solution

seed, item_n, id = Random_select_function(X, item_id, item_len) # 0:solution value 1: solution weight item_n: solution numbers # kokomokakuninn
# print seed

#Call the Functionguided_search(X, city_tour = seed, alpha = 0.5, local_search_optima = 100, max_attempts = 10, iterations = 1000) #change iterationsnumber to 10 from 2500
guided_search(X, seed, id, item_n, item_len, max_attempts = 10, iterations = 1000) #change iterationsnumber to 10 from 2500

