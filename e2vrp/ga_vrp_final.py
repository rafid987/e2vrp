
# Required Libraries
import folium
import folium.plugins
import pandas as pd
import random
import numpy as np
import copy
import os
import time as tm

from itertools import cycle
from matplotlib import pyplot as plt
plt.style.use('bmh')


def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Subroute Distance
def evaluate_distance(distance_matrix, satellite, subroute):    
    subroute_i    = satellite + subroute
    subroute_j    = subroute + satellite
    subroute_ij   = [(subroute_i[i], subroute_j[i]) for i in range(0, len(subroute_i))]
    distance      = list(np.cumsum(distance_matrix[tuple(np.array(subroute_ij).T)]))
    distance[0:0] = [0.0]
    return distance

# Function: Subroute Capacity
def evaluate_capacity(parameters, satellite, subroute): 
    demand    = parameters[:]
    subroute_ = satellite + subroute + satellite
    capacity  = list(np.cumsum(demand[subroute_]))
    return capacity 

# Function: Subroute Cost
def evaluate_cost(dist, parameters, satellite, subroute):
    
    subroute_ = satellite + subroute + satellite
    cost      = [0]*len(subroute_)  
    cost = [0 if x == 0 else 0 + x  for x in dist]
    return cost

# Function: Subroute Cost
def evaluate_cost_penalty(dist, cap, capacity, parameters, satellite, subroute, penalty_value):

    subroute_ = satellite + subroute + satellite
    pnlt = 0
    cost = [0]*len(subroute_)
    pnlt = pnlt + sum( x > capacity for x in cap[0:len(subroute_)] )
    
    cost = [0 if x == 0 else cost[0] + x for x in dist]        
    cost[-1] = cost[-1] + pnlt*penalty_value
    return cost[-1]

# Function: Routes Nearest satellite
def evaluate_satellite(satellites, individual, distance_matrix):
    d_1 = float('+inf')
    for i in range(0, satellites):
        for j in range(0, len(individual[1])):
            d_2 = evaluate_distance(distance_matrix, [i], individual[1][j])[-1]
            if (d_2 < d_1):
                d_1 = d_2
                individual[0][j] = [i]
    return individual

# Function: Routes Break Capacity
def cap_break(vehicle_types, individual, parameters, capacity):
    go_on = True
    while (go_on):
        individual_ = copy.deepcopy(individual)
        solution    = [[], [], []]
        for i in range(0, len(individual_[0])):
            cap   = evaluate_capacity(parameters, individual_[0][i], individual_[1][i]) 
            sep   = [x >  capacity[individual_[2][i][0]] for x in cap[1:-1] ]
            sep_f = [individual_[1][i][x] for x in range(0, len(individual_[1][i])) if sep[x] == False]
            sep_t = [individual_[1][i][x] for x in range(0, len(individual_[1][i])) if sep[x] == True ]
            if (len(sep_t) > 0 and len(sep_f) > 0):
                solution[0].append(individual_[0][i])
                solution[0].append(individual_[0][i])
                solution[1].append(sep_f)
                solution[1].append(sep_t)
                solution[2].append(individual_[2][i])
                solution[2].append(individual_[2][i])
            if (len(sep_t) > 0 and len(sep_f) == 0):
                solution[0].append(individual_[0][i])
                solution[1].append(sep_t)
                solution[2].append(individual_[2][i])
            if (len(sep_t) == 0 and len(sep_f) > 0):
                solution[0].append(individual_[0][i])
                solution[1].append(sep_f)
                solution[2].append(individual_[2][i])
        individual_ = copy.deepcopy(solution)
        if (individual == individual_):
            go_on      = False
        else:
            go_on      = True
            individual = copy.deepcopy(solution)
    return individual

def e2_cost(sol, satellites, customer, distance_matrix, demand):
  v2_cap=12500
  cap=[0]*3
  for i in range(0,len(sol[0])):
    cap[sol[0][i][0]]=cap[sol[0][i][0]]+sum(demand[sol[1][i]])
  #print(cap)
  #cap=[1600, 1900, 1300]
  e2cost=0
  depot=customer+satellites
  for i in range(satellites):
    while cap[i]>v2_cap:
      cap[i]=cap[i]-v2_cap
      e2cost=e2cost+ 2*distance_matrix[depot][i+1]
      #print(e2cost)
  #print(cap)
  l_cap=0
  e2cost1=0
  subroute=[]
  c=0
  for i in range(satellites):
    e2cost2=0
    l_cap=0
    if cap[i]>0:
      if l_cap+cap[i]<=v2_cap:
        l_cap=l_cap+cap[i]
        subroute.append([])
        subroute[c]=subroute[c]+[i+1]
      for j in range(i+1,satellites):
        if cap[j]>0:
          if l_cap+cap[j]<=v2_cap:
            l_cap=l_cap+cap[j]
            cap[j]=0
            subroute[c]=subroute[c]+[j+1]               
    elif cap[i]==0:
      continue
    e2cost1=e2cost1+ evaluate_distance(distance_matrix, [depot], subroute[c])[-1]
    c+=1
  #print(subroute)
  e2cost=e2cost+e2cost1
  return e2cost

# Function: Route Evalution & Correction
def target_function(population, distance_matrix, satellites, customer, parameters, capacity, penalty_value, fleet_size = []):
    cost     = [[0] for i in range(len(population))]
    flt_cnt  = [0]*len(fleet_size)

    for k in range(0, len(population)): # k individuals
        individual = copy.deepcopy(population[k])  
        e2cost     = e2_cost(individual, satellites, customer, distance_matrix, parameters)
        size       = len(individual[1])
        i          = 0
        pnlt       = 0
        flt_cnt    = [0]*len(fleet_size)
        while (size > i): # i subroutes 
            dist = evaluate_distance(distance_matrix, individual[0][i], individual[1][i])
            
            cap    = evaluate_capacity(parameters, satellite = individual[0][i], subroute = individual[1][i])
            cost_s = evaluate_cost(dist, parameters, satellite = individual[0][i], subroute = individual[1][i])      
            pnlt   = pnlt + sum( x >  capacity[individual[2][i][0]] for x in cap[0:-1] )
                               
            if (len(fleet_size) > 0):
                flt_cnt[individual[2][i][0]] = flt_cnt[individual[2][i][0]] + 1 
            if (size <= i + 1):
                for v in range(0, len(fleet_size)):
                    v_sum = flt_cnt[v] - fleet_size[v]
                    if (v_sum > 0):
                        pnlt = pnlt + v_sum
            cost[k][0] = cost[k][0] + cost_s[-1] + pnlt*penalty_value
            cost[k][0] = cost[k][0] + e2cost
            size       = len(individual[1])
            i          = i + 1
    cost_total = copy.deepcopy(cost)
    return cost_total, population

# Function: Initial Population
def initial_population(distance_matrix = 'none', population_size = 5, vehicle_types = 1, n_satellites = 1):
    
    satellites     = [[i] for i in range(0, n_satellites)]
    vehicles   = [[i] for i in range(0, vehicle_types)]
    clients    = list(range(n_satellites, 53))
    population = []
    for i in range(0, population_size):
        clients_temp    = copy.deepcopy(clients)
        routes          = []
        routes_satellite    = []
        routes_vehicles = []
        while (len(clients_temp) > 0):
            e = random.sample(vehicles, 1)[0]
            d = random.sample(satellites, 1)[0]
            
            c = random.sample(clients_temp, random.randint(1, len(clients_temp)))
            routes_vehicles.append(e)
            routes_satellite.append(d)
            routes.append(c)
            clients_temp = [item for item in clients_temp if item not in c]
        population.append([routes_satellite, routes, routes_vehicles])
    return population

# Function: Fitness
def fitness_function(cost, population_size): 
    fitness = np.zeros((population_size, 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1 + cost[i][0] + abs(np.min(cost)))
    fit_sum      = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    return fitness

# Function: Selection
def roulette_wheel(fitness): 
    ix     = 0
    random = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix


def crossover1(parent_1, parent_2):
    s         = random.sample(list(range(0,len(parent_1[0]))), 1)[0]
    subroute  = [ parent_1[0][s], parent_1[1][s], parent_1[2][s] ]
    offspring = copy.deepcopy(parent_2)
    for k in range(len(parent_2[1])-1, -1, -1):
        offspring[1][k] = [item for item in offspring[1][k] if item not in subroute[1] ] 
        if (len(offspring[1][k]) == 0):
            del offspring[0][k]
            del offspring[1][k]
            del offspring[2][k]
    offspring[0].append(subroute[0])
    offspring[1].append(subroute[1])
    offspring[2].append(subroute[2])
    return offspring


def crossover2(parent_1, parent_2, distance_matrix, capacity, penalty_value, parameters):
    s         = random.sample(list(range(0,len(parent_1[0]))), 1)[0]
    offspring = copy.deepcopy(parent_2)
    if (len(parent_1[1][s]) > 1):
        cut  = random.sample(list(range(0,len(parent_1[1][s]))), 2)
        gene = 2
    else:
        cut  = [0, 0]
        gene = 1
    for i in range(0, gene):
        d_1   = float('+inf')
        ins_m = 0
        A     = parent_1[1][s][cut[i]]
        best  = []
        for m in range(0, len(parent_2[1])):
            parent_2[1][m] = [item for item in parent_2[1][m] if item not in [A] ]
            if (len(parent_2[1][m]) > 0):
                insertion      = copy.deepcopy([ parent_2[0][m], parent_2[1][m], parent_2[2][m] ])
                dist_list      = [evaluate_distance(distance_matrix, insertion[0], insertion[1][:n] + [A] + insertion[1][n:]) for n in range(0, len(parent_2[1][m]) + 1)]
                wait_time_list = [[0, 0]]*len(dist_list)
                cap_list       = [evaluate_capacity(parameters, insertion[0], insertion[1][:n] + [A] + insertion[1][n:]) for n in range(0, len(parent_2[1][m]) + 1)]
                insertion_list = [insertion[1][:n] + [A] + insertion[1][n:] for n in range(0, len(parent_2[1][m]) + 1)]
                d_2_list       = [evaluate_cost_penalty(dist_list[n], cap_list[n], capacity[parent_2[2][m][0]], parameters, insertion[0], insertion_list[n], penalty_value) for n in range(0, len(dist_list))]
                d_2 = min(d_2_list)
                if (d_2 <= d_1):
                    d_1   = d_2
                    ins_m = m
                    best  = insertion_list[d_2_list.index(min(d_2_list))]
        parent_2[1][ins_m] = best            
        if (d_1 != float('+inf')):
            offspring = copy.deepcopy(parent_2)
    for i in range(len(offspring[1])-1, -1, -1):
        if(len(offspring[1][i]) == 0):
            del offspring[0][i]
            del offspring[1][i]
            del offspring[2][i]
    return offspring

# Function: Breeding
def breeding(cost, population, fitness, distance_matrix, satellites, elite, capacity, penalty_value, parameters, vehicle_types, fleet_size):
    offspring = copy.deepcopy(population) 
    if (elite > 0):
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        for i in range(0, elite):
            offspring[i] = copy.deepcopy(population[i])
    for i in range (elite, len(offspring)):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
        parent_1 = copy.deepcopy(population[parent_1])  
        parent_2 = copy.deepcopy(population[parent_2])
        rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)  
        
        
        if((len(parent_1[1]) > 1 and len(parent_2[1]) > 1)):
            if (rand > 0.5):
                offspring[i] = crossover1(parent_1, parent_2)
                offspring[i] = crossover2(offspring[i], parent_2, distance_matrix, capacity, penalty_value, parameters = parameters)              
            elif (rand <= 0.5): 
                offspring[i] = crossover1(parent_2, parent_1)
                offspring[i] = crossover2(offspring[i], parent_1, distance_matrix, capacity, penalty_value, parameters = parameters)
        if (satellites > 1):
            offspring[i] = evaluate_satellite(satellites, offspring[i], distance_matrix) 
        
        offspring[i] = cap_break(vehicle_types, offspring[i], parameters, capacity)
    return offspring

# Function: Mutation - Swap
def mutation_swap(individual):
    if (len(individual[1]) == 1):
        k1 = random.sample(list(range(0, len(individual[1]))), 1)[0]
        k2 = k1
    else:
        k  = random.sample(list(range(0, len(individual[1]))), 2)
        k1 = k[0]
        k2 = k[1]  
    cut1                    = random.sample(list(range(0, len(individual[1][k1]))), 1)[0]
    cut2                    = random.sample(list(range(0, len(individual[1][k2]))), 1)[0]
    A                       = individual[1][k1][cut1]
    B                       = individual[1][k2][cut2]
    individual[1][k1][cut1] = B
    individual[1][k2][cut2] = A
    return individual

# Function: Mutation - Insertion
def mutation_insertion(individual):
    if (len(individual[1]) == 1):
        k1 = random.sample(list(range(0, len(individual[1]))), 1)[0]
        k2 = k1
    else:
        k  = random.sample(list(range(0, len(individual[1]))), 2)
        k1 = k[0]
        k2 = k[1]
    cut1 = random.sample(list(range(0, len(individual[1][k1])))  , 1)[0]
    cut2 = random.sample(list(range(0, len(individual[1][k2])+1)), 1)[0]
    A    = individual[1][k1][cut1]
    del individual[1][k1][cut1]
    individual[1][k2][cut2:cut2] = [A]
    if (len(individual[1][k1]) == 0):
        del individual[0][k1]
        del individual[1][k1]
        del individual[2][k1]
    return individual

# Function: Mutation
def mutation(offspring, mutation_rate, elite):
    for i in range(elite, len(offspring)):
        probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
        if (probability <= mutation_rate):
            rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            if (rand <= 0.5):
                offspring[i] = mutation_insertion(offspring[i])
            elif(rand > 0.5):
                offspring[i] = mutation_swap(offspring[i])
        for k in range(0, len(offspring[i][1])):
            if (len(offspring[i][1][k]) >= 2):
                probability = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                if (probability <= mutation_rate):
                    rand = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    cut  = random.sample(list(range(0, len(offspring[i][1][k]))), 2)
                    cut.sort()
                    C    = offspring[i][1][k][cut[0]:cut[1]+1]
                    if (rand <= 0.5):
                        random.shuffle(C)
                    elif(rand > 0.5):
                        C.reverse()
                    offspring[i][1][k][cut[0]:cut[1]+1] = C
    return offspring

# Function: Elite Distance
def elite_distance(individual, distance_matrix):

    td = 0
    for n in range(0, len(individual[1])):
        td = td + evaluate_distance(distance_matrix, satellite = individual[0][n], subroute = individual[1][n])[-1]
    return round(td,2)


def genetic_algorithm_solo(distance_matrix, satellites, customer, parameters, capacity, population_size = 5, vehicle_types = 1, fleet_size = [], mutation_rate = 0.1, elite = 0, generations = 50, penalty_value = 1000):    
    start           = tm.time()
    count           = 0
    
    max_capacity    = copy.deepcopy(capacity)
     
    # for i in range(0, satellites):
    #     parameters[i] = 0  
    population       = initial_population(distance_matrix, population_size = population_size, vehicle_types = vehicle_types, n_satellites = satellites)

    cost, population = target_function(population, distance_matrix, satellites, customer, parameters, max_capacity, penalty_value, fleet_size = fleet_size) 
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    
    fitness          = fitness_function(cost, population_size)
    
    elite_ind        = elite_distance(population[0], distance_matrix)
    cost             = copy.deepcopy(cost)
    elite_cst        = copy.deepcopy(cost[0][0])
    solution         = copy.deepcopy(population[0])
    print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2)) 

    while (count <= generations-1): 
        offspring        = breeding(cost, population, fitness, distance_matrix, satellites, elite, max_capacity, penalty_value, parameters, vehicle_types, fleet_size)   
        offspring        = mutation(offspring, mutation_rate = mutation_rate, elite = elite)
        cost, population = target_function(offspring, distance_matrix, satellites, customer, parameters, max_capacity, penalty_value, fleet_size = fleet_size)
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        elite_child      = elite_distance(population[0], distance_matrix)
        
        fitness = fitness_function(cost, population_size)

        if(elite_ind > elite_child):
            elite_ind = elite_child 
            solution  = copy.deepcopy(population[0])
            elite_cst = copy.deepcopy(cost[0][0])
        count = count + 1  
        print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2))
    
    end = tm.time()
    print('Algorithm Time: ', round((end - start), 2), ' seconds')
    return solution, population
   
   ############################################################################

def genetic_algorithm_swarmp(init_pop, distance_matrix,  satellites, customers, parameters, capacity, population_size = 5, vehicle_types = 1, fleet_size = [], mutation_rate = 0.1, elite = 0, generations = 50, penalty_value = 1000):    
    start           = tm.time()
    count           = 0
    
    max_capacity    = copy.deepcopy(capacity)
     
    # for i in range(0, satellites):
    #     parameters[i] = 0  
    population       = copy.deepcopy(init_pop)

    cost, population = target_function(population, distance_matrix, satellites, customers, parameters, max_capacity, penalty_value, fleet_size = fleet_size) 
    cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
    
    fitness          = fitness_function(cost, population_size)
    
    elite_ind        = elite_distance(population[0], distance_matrix)
    cost             = copy.deepcopy(cost)
    elite_cst        = copy.deepcopy(cost[0][0])
    solution         = copy.deepcopy(population[0])
    print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2)) 

    while (count <= generations-1): 
        offspring        = breeding(cost, population, fitness, distance_matrix, satellites, elite, max_capacity, penalty_value, parameters, vehicle_types, fleet_size)   
        offspring        = mutation(offspring, mutation_rate = mutation_rate, elite = elite)
        cost, population = target_function(offspring, distance_matrix, satellites, customers, parameters, max_capacity, penalty_value, fleet_size = fleet_size)
        cost, population = (list(t) for t in zip(*sorted(zip(cost, population))))
        elite_child      = elite_distance(population[0], distance_matrix)
        
        fitness = fitness_function(cost, population_size)

        if(elite_ind > elite_child):
            elite_ind = elite_child 
            solution  = copy.deepcopy(population[0])
            elite_cst = copy.deepcopy(cost[0][0])
        count = count + 1  
        print('Generation = ', count, ' Distance = ', elite_ind, ' f(x) = ', round(elite_cst, 2))
    
    end = tm.time()
    print('Algorithm Time: ', round((end - start), 2), ' seconds')
    return solution, population
   
   ############################################################################
