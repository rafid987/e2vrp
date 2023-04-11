
# Required Libraries
import numpy  as np
import math
import random
import os
from .utils_f import  *
############################################################################

############################################################################

# Function: Initialize Variables
def initial_position(hunting_party = 5, n_dimentions=20, min_values = [0], max_values = [60]):
    position = np.zeros((hunting_party, n_dimentions+1))
    for i in range(0, hunting_party):
        tem=[random.random() * random.uniform(min_values[0], max_values[0])  for i in range(n_dimentions*2)]
        tem=random.sample(tem, n_dimentions)

        for j in range(0, n_dimentions):
             position[i,j] = tem[j]
    return position


def update_fitness(position, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size = [], min_values = [0], max_values = [60]):
    for i in range(0, position.shape[0]):
        population1=decoderpso(satellites, position[i,0:position.shape[1]-1])
        fitness= target_function_single(population1, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size = [])
        position[i,-1] = fitness[0]
    return position

############################################################################

# Function: Initialize Alpha
def leader_position(dimension = 20):
    leader = np.zeros((1, dimension+1))
    for j in range(0, dimension):
        leader[0,j] = 0.0
    #leader[0,-1] = target_function(leader[0,0:leader.shape[1]-1])
    leader[0,-1] = 9999999999
    return leader

def update_leader(position, leader):
    updated_position = np.copy(position)
    for i in range(0, position.shape[0]):
        if (updated_position[i,-1] < leader[0,-1]):
            leader[0,:] = np.copy(updated_position[i,:])
    return leader

def reflect(x, lower, upper, **_kwargs):
    r"""Repair solution and put the solution in search space with reflection of how much the solution violates a bound.
    Args:
        x (numpy.ndarray): Solution to be fixed.
        lower (numpy.ndarray): Lower bounds of search space.
        upper (numpy.ndarray): Upper bounds of search space.
    Returns:
        numpy.ndarray: Fix solution.
    """
    if x > upper:
        x = lower + x % (upper - lower)
    elif x < lower:
        x = lower + x % (upper - lower)
    return x

# Function: Updtade Position
def update_position(position, leader, a_linear_component = 2, b_linear_component = 1,  spiral_param = 1, min_values = [0], max_values = [60]): 
    updated_position = np.copy(position)

    for i in range(0, updated_position.shape[0]):

            r1_leader = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            r2_leader = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            a_leader  = 2*a_linear_component*r1_leader - a_linear_component
            c_leader  = 2*r2_leader           
            p         = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)     

            for j in range (0, updated_position.shape[1] - 1):
                if (p < 0.5):
                    if (abs(a_leader) >= 1):
                        rand              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                        rand_leader_index = math.floor(updated_position.shape[0]*rand);
                        x_rand            = updated_position[rand_leader_index, :]
                        distance_x_rand   = abs(c_leader*x_rand[j] - updated_position[i,j]) 

                        updated_position[i,j]     = reflect( x_rand[j] - a_leader*distance_x_rand, min_values[0],  max_values[0])                                         
                    elif (abs(a_leader) < 1):
                        distance_leader = abs(c_leader*leader[0,j] - updated_position[i,j]) 
                        updated_position[i,j]   = reflect(leader[0,j] - a_leader*distance_leader, min_values[0],  max_values[0])       

                elif (p >= 0.5):      
                    distance_Leader = abs(leader[0,j] - updated_position[i,j])
                    rand            = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
                    m_param         = (b_linear_component - 1)*rand + 1
                    updated_position[i,j]   = reflect( (distance_Leader*math.exp(spiral_param*m_param)*math.cos(m_param*2*math.pi) + leader[0,j]), min_values[0],  max_values[0])              
            #updated_position[i,-1] = target_function(updated_position[i,0:updated_position.shape[1]-1])           
    return updated_position
    

############################################################################

# WOA Function
def whale_optimization_algorithm(hunting_party, spiral_param, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size = [],  min_values = [0], max_values = [60], iterations = 50, verbose = True):    
    count    = 0
    leader   = leader_position(dimension = customers)
    position = initial_position(hunting_party = hunting_party,  n_dimentions=customers,  min_values = min_values, max_values = max_values)
    position = update_fitness(position, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size, min_values, max_values)
    leader   = update_leader(position, leader)
    print('Iteration = ', count, ' f(x) = ', leader[0][-1])

    while (count <= iterations):
        
        a_linear_component =  2 - count*( 2/iterations)
        b_linear_component = -1 + count*(-1/iterations)

        position = update_position(position, leader, a_linear_component, b_linear_component,  spiral_param, min_values = min_values, max_values = max_values)    
        position = update_fitness(position, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size, min_values, max_values)
        
        leader             = update_leader(position, leader)       

        count              = count + 1   
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', leader[0,-1])
    return leader, position


############################################################################
