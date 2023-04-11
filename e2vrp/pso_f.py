# Required Libraries
import numpy  as np
import random
import os
from .utils_f import  *
############################################################################


############################################################################

# Function: Initialize Variables
def initial_position(swarm_size = 3, n_dimentions=20, min_values = [0], max_values = [60]):
    position = np.zeros((swarm_size, n_dimentions+1))
    for i in range(0, swarm_size):
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

# Function: Initialize Velocity
def initial_velocity(position, n_dimentions, min_values = [-5], max_values = [5]):
    init_velocity = np.zeros((position.shape[0], n_dimentions))
    for i in range(0, init_velocity.shape[0]):
        for j in range(0, init_velocity.shape[1]):
            init_velocity[i,j] = random.uniform(min_values[0], max_values[0])
    return init_velocity

# Function: Individual Best
def individual_best_matrix(position, i_b_matrix): 
    for i in range(0, position.shape[0]):
        if(i_b_matrix[i,-1] > position[i,-1]):
            for j in range(0, position.shape[1]):
                i_b_matrix[i,j] = position[i,j]
    return i_b_matrix

# Function: Velocity
def velocity_vector(position, init_velocity, i_b_matrix, best_global, w = 0.5, c1 = 2, c2 = 2):
    r1       = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    r2       = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    velocity = np.zeros((position.shape[0], init_velocity.shape[1]))
    for i in range(0, init_velocity.shape[0]):
        for j in range(0, init_velocity.shape[1]):
            velocity[i,j] = w*init_velocity[i,j] + c1*r1*(i_b_matrix[i,j] - position[i,j]) + c2*r2*(best_global[j] - position[i,j])
    return velocity

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
def update_position(position, velocity, min_values = [0], max_values = [60]):
    for i in range(0, position.shape[0]):
        for j in range(0, position.shape[1] - 1):
            #position[i,j] = np.clip((position[i,j] + velocity[i,j]),  min_values[j],  max_values[j])
            position[i,j] = reflect((position[i,j] + velocity[i,j]),  min_values[0],  max_values[0])

        #position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

############################################################################

# PSO Function
def particle_swarm_optimization(swarm_size, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size = [], min_values = [0], max_values = [60], iterations = 50, decay = 0, w = 0.9, c1 = 2, c2 = 2, verbose = True):    
    count         = 0
    position      = initial_position(swarm_size, customers, min_values, max_values)
    position      = update_fitness(position, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size, min_values, max_values)
    init_velocity = initial_velocity(position,customers, min_values, max_values)
    i_b_matrix    = np.copy(position)
    best_global   = np.copy(position[position[:,-1].argsort()][0,:])
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', best_global[-1])
        position    = update_position(position, init_velocity, min_values, max_values)
        position    = update_fitness(position, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size, min_values, max_values)            
        i_b_matrix  = individual_best_matrix(position, i_b_matrix)
        value       = np.copy(i_b_matrix[i_b_matrix[:,-1].argsort()][0,:])
        if (best_global[-1] > value[-1]):
            best_global = np.copy(value)   
        if (decay > 0):
            n  = decay
            w  = w*(1 - ((count-1)**n)/(iterations**n))
            c1 = (1-c1)*(count/iterations) + c1
            c2 = (1-c2)*(count/iterations) + c2
        init_velocity = velocity_vector(position, init_velocity, i_b_matrix, best_global, w = w, c1 = c1, c2 = c2)
        count         = count + 1     
    return best_global, position

############################################################################
