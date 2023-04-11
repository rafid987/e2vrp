
# Required Libraries
import pandas as pd
import random
import numpy as np
import copy
import os
import time as tm
from itertools import cycle

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


def target_function_single(population1, distance_matrix, satellites, customer, parameters, capacity, penalty_value, fleet_size = []):
    cost     = [0]
    flt_cnt  = [0]*len(fleet_size)

    individual = copy.deepcopy(population1)
    e2cost     = e2_cost(individual, satellites, customer, distance_matrix, parameters)  
    size       = len(individual[1])
    i          = 0
    pnlt       = 0
    flt_cnt    = [0]*len(fleet_size)
    while (size > i): # i subroutes 
        dist = evaluate_distance(distance_matrix, individual[0][i], individual[1][i])
        cap    = evaluate_capacity(parameters, satellite = individual[0][i], subroute = individual[1][i])
        cost_s = evaluate_cost(dist, parameters, satellite = individual[0][i], subroute = individual[1][i])     
        pnlt   = pnlt + sum( x >  capacity[0] for x in cap[0:-1] )

        if (len(fleet_size) > 0):
            flt_cnt[individual[2][i][0]] = flt_cnt[individual[2][i][0]] + 1 
        if (size <= i + 1):
            for v in range(0, len(fleet_size)):
                v_sum = flt_cnt[v] - fleet_size[v]
                if (v_sum > 0):
                    pnlt = pnlt + v_sum                    
        cost[0] = cost[0] + cost_s[-1] + pnlt*penalty_value
        cost[0] = cost[0] + e2cost
        size       = len(individual[1])
        i          = i + 1
    cost_total = copy.deepcopy(cost)
    return cost_total

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

def decoderpso(satellites,genotype):
  sol=[[] for i in range(satellites)]
  divider=60/satellites
  scale = [0]
  for i in range(satellites):
      scale.append(scale[-1] + divider)
  #scale
  ordered = (np.argsort(genotype)+satellites).tolist()
  for i in range(len(genotype)):
    for j in range(len(scale)-1):
        if genotype[ordered[i]-satellites] == scale[-1]:
            sol[-1].append(ordered[i])
            break

        if scale[j] <= genotype[ordered[i]-satellites] < scale[j + 1]:
            sol[j].append(ordered[i])
            break
  #print(sol)
  k=0
  sol1=[]
  dep=[]
  tem=[]
  #sol1.append([])
  for i in range(satellites):
      for j in range(len(sol[i])):        
        if j+1<len(sol[i]):
          if genotype[sol[i][j+1]-satellites]-genotype[sol[i][j]-satellites]>5:
              tem.append(sol[i][j])
              sol1.append(tem)
              dep.append([i])
              tem=[]
              #tem.append(sol[i][j])
          else:
            tem.append(sol[i][j])            
        else:
          tem.append(sol[i][j])
      if tem!=[]:
        sol1.append(tem)
        dep.append([i])
        tem=[]
  
  f_sol=[dep, sol1, [[0] for i in range(len(sol1))]]
  return f_sol

def decode_pop(position, satellites):
    population=[]
    for i in range(0, position.shape[0]):
        ind_pop = decoderpso(satellites, position[i,0:position.shape[1]-1])
        population.append(ind_pop)
    return population

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
  
