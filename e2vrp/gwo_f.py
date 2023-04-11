
# Required Libraries
import numpy  as np
import random
import os
from .utils_f import  *
############################################################################

############################################################################

# Function: Initialize Variables
def initial_position(pack_size = 5, n_dimentions=20, min_values = [0], max_values = [60]):
    position = np.zeros((pack_size, n_dimentions+1))
    for i in range(0, pack_size):
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
def alpha_position(dimension = 20):
    alpha = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        alpha[0,j] = 0.0
    #alpha[0,-1] = target_function(alpha[0,0:alpha.shape[1]-1])
    alpha[0,-1] = 9999999999
    return alpha

# Function: Initialize Beta
def beta_position(dimension = 20):
    beta = np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        beta[0,j] = 0.0
    #beta[0,-1] = target_function(beta[0,0:beta.shape[1]-1])
    beta[0,-1] = 9999999999
    return beta

# Function: Initialize Delta
def delta_position(dimension = 20):
    delta =  np.zeros((1, dimension + 1))
    for j in range(0, dimension):
        delta[0,j] = 0.0
    #delta[0,-1] = target_function(delta[0,0:delta.shape[1]-1])
    delta[0,-1] = 9999999999
    return delta

# Function: Updtade Pack by Fitness
def update_pack(position, alpha, beta, delta):
    updated_position = np.copy(position)
    for i in range(0, position.shape[0]):
        if (updated_position[i,-1] < alpha[0,-1]):
            alpha[0,:] = np.copy(updated_position[i,:])
        elif (alpha[0,-1] < updated_position[i,-1] < beta[0,-1]):          
            beta[0,:] = np.copy(updated_position[i,:])
        elif (beta[0,-1]  < updated_position[i,-1] < delta[0,-1]):
            delta[0,:] = np.copy(updated_position[i,:])
    return alpha, beta, delta

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
def update_position(position, alpha, beta, delta, a_linear_component = 2, min_values = [0], max_values = [60]):
    updated_position = np.copy(position)
    for i in range(0, updated_position.shape[0]):
        for j in range(0, position.shape[1] - 1):   
            r1_alpha              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            r2_alpha              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            a_alpha               = 2*a_linear_component*r1_alpha - a_linear_component
            c_alpha               = 2*r2_alpha      
            distance_alpha        = abs(c_alpha*alpha[0,j] - position[i,j]) 
            x1                    = alpha[0,j] - a_alpha*distance_alpha   
            r1_beta               = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            r2_beta               = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            a_beta                = 2*a_linear_component*r1_beta - a_linear_component
            c_beta                = 2*r2_beta            
            distance_beta         = abs(c_beta*beta[0,j] - position[i,j]) 
            x2                    = beta[0,j] - a_beta*distance_beta                          
            r1_delta              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            r2_delta              = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
            a_delta               = 2*a_linear_component*r1_delta - a_linear_component
            c_delta               = 2*r2_delta            
            distance_delta        = abs(c_delta*delta[0,j] - position[i,j]) 
            x3                    = delta[0,j] - a_delta*distance_delta                                 
            updated_position[i,j] = reflect(((x1 + x2 + x3)/3),min_values[0],max_values[0])   
        #updated_position[i,-1] = target_function(updated_position[i,0:updated_position.shape[1]-1])
    return updated_position

############################################################################

# GWO Function
def grey_wolf_optimizer(pack_size, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size = [], min_values = [0], max_values = [60], iterations = 50, verbose = True):      
    count    = 0
    alpha    = alpha_position(dimension = customers)
    beta     = beta_position(dimension  = customers)
    delta    = delta_position(dimension = customers)
    position = initial_position(pack_size = pack_size,  n_dimentions=customers,  min_values = min_values, max_values = max_values)
    position = update_fitness(position, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size, min_values, max_values)
    alpha, beta, delta = update_pack(position, alpha, beta, delta)
    print('Iteration = ', count, ' f(x) = ', alpha[0][-1])

    while (count <= iterations): 
        
        a_linear_component = 2 - count*(2/iterations)

        position = update_position(position, alpha, beta, delta, a_linear_component = a_linear_component, min_values = min_values, max_values = max_values)    
        position = update_fitness(position, distance_matrix, satellites, customers, parameters, capacity, penalty_value, fleet_size, min_values, max_values)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        
        count              = count + 1          
        if (verbose == True):    
            print('Iteration = ', count, ' f(x) = ', alpha[0][-1])      
    return alpha, position

############################################################################
