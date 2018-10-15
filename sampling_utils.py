#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 13:11:43 2018

@author: dorran
"""

import pystan
import pickle
import os
import numpy as np

def create_model(filename):
    """
        Compiles a stan model using a text file with the model definition
    """
    cpp_compiler_args = [
                '-O3',
                '-ftemplate-depth-256',
                '-Wno-unused-function',
                '-Wno-uninitialized']
    
    model = pystan.StanModel(file=filename,verbose=True,
                             extra_compile_args=cpp_compiler_args)
    
    file_path_parts = list(os.path.split(filename))
    file_path_parts[-1] = file_path_parts[-1].split(".")[0] + ".pkl"
    pickle_filename = os.path.join(*file_path_parts)
    with open(pickle_filename,'wb') as f:
        pickle.dump(model, f)
    
    return model,pickle_filename

def load_model(pickle_filename):
    with open(pickle_filename,'rb') as f:    
        model = pickle.load(f)
    return model



# Ported from MATLAB
def row_fill(A, i, m, n, receiver, chi, dt, row_global_map, global_row_map):
    q = i
    itime = 1
            
    global_time = chi[q]
    local_time = dt
    filled_row = np.zeros((1, (n * m)))

    while global_time > 0.00001:
        i_rec = receiver[row_global_map[q]]

        next_tau = chi[global_row_map[i_rec]]

        j = n * (itime - 1) + q

        if (global_time - next_tau) >= local_time:
            filled_row[0,j] = local_time
            global_time = global_time - local_time
            itime = itime + 1
            local_time = dt
        else:
            filled_row[0,j] = global_time - next_tau
            local_time = local_time - (global_time - next_tau)
            global_time = next_tau
            # Row of the receiver
            q = global_row_map[i_rec]
                
    A[i, :] = filled_row
    return

def prepare_model_data(topo_datafile,M,sigmaData,sigmaM,meanM,minM,maxM):
    # Load data from txt file
    # elev, chi, x , y
    data_array = np.loadtxt(topo_datafile, delimiter=',', skiprows=1, usecols=[2, 3, 4, 5])
    # pt, receiver, j
    indexing_array = np.loadtxt(topo_datafile, dtype='int', delimiter=',', skiprows=1, usecols=[0, 1, 6])
    
    # Get number of nodes
    N = np.size(data_array, 0)
    ndim = N*M
    
    # Create mapping indices
    row_to_global = {}
    global_to_row = {}
    receiver = {}
    for i in range(0, N):
        n = indexing_array[i, 0]
        r = indexing_array[i, 2] - 1
        rec = indexing_array[i, 1]
        row_to_global[r] = n
        global_to_row[n] = r
        receiver[n] = rec
    
    chi = data_array[:, 1]
    
    dt = np.max(chi) / M
    
    
    # Initialize Model Matrix - Using a LIL sparse for population then 
    # later converting it to CSR format for serialization and improved
    # performance
    A = np.matrix(np.zeros((N, M*N), dtype='float'))
    
    # Fill Model Matrix
    print("--> Processing: Populating Forward Model Matrix...")
    for i in range(1, N): #N
        row_fill(A, i, M, N, receiver, chi, dt, row_to_global, global_to_row)
        if (i % 1000 == 0):
            print("Node: {0} / {1}".format(i,N))
    
    # d0 - data observations
    d = data_array[:,0]
    d0 = np.zeros([N])
    d0[:] = d
    minElev = d0.min()
    d0 = d0-minElev
    
    input_data = {
            "ndim":ndim,
            "N":N,
            "M":M,
            "data_array":data_array,
            "indexing_array":indexing_array,
            "d0":d0,
            "minElev":minElev,
            "A":A,
            "row_to_global":row_to_global,
            "global_to_row":global_to_row,
            "dt":dt
            }

    return input_data    
    
def spatial_temporal_correlation_cov(input_data,L,T,sigma,dist_cor_func=None,time_cor_func=None):
    """
        Populate a covariance matrix for model parameters:
            * Assumes 0 temporal correlation
    """
    import math
    def dist(x1,y1,x2,y2):
        return math.sqrt((x2-x1)**2.0 + (y2-y1)**2.0)
    
    def default_spatial_correlation_func(d,L):
        C = np.exp((-1*d)/L)
        return C
    
    def default_time_correlation_func(t1,t2,T):
        C = np.exp((-1*np.abs(t2-t1)) / T )
        return C
    
    if dist_cor_func is None:
        dist_cor_func = default_spatial_correlation_func
    if time_cor_func is None:
        time_cor_func = default_time_correlation_func
    
    
    x = input_data['data_array'][:,2]
    y = input_data['data_array'][:,3]
    
    # Initialize Covariance Matrix (Dense)
    cm = np.zeros([input_data['ndim'],input_data['ndim']])
    
    rowN = 0 # Tracks spatial index of the row
    # For each row
    for r in range(0,input_data["ndim"]):
        rowM = math.floor(r/input_data["N"])
        rowX = x[rowN]
        rowY = y[rowN]
        rowt = rowM*input_data["dt"]
        
        for m in range(0,input_data["M"]):
            for n in range(0,input_data["N"]):
                c = m*input_data["N"] + n
                thisX = x[n]
                thisY = y[n]
                thist = m*input_data["dt"]
                d = math.sqrt((thisX-rowX)**2.0 + (thisY-rowY)**2.0)
                dist_cor = dist_cor_func(d,L)
                temp_cor = time_cor_func(rowt,thist,T)
                #print(temp_cor)
                cm[r,c] = (sigma**2.0) * dist_cor * temp_cor
        if rowN == input_data["N"]-1:
            rowN = 0;
        else:
            rowN = rowN+1
    return cm
    
def spatial_correlation_cov(input_data,L,sigma,cor_func=None):
    """
        Populate a covariance matrix for model parameters:
            * Assumes 0 temporal correlation
    """
    import math
    def dist(x1,y1,x2,y2):
        return math.sqrt((x2-x1)**2.0 + (y2-y1)**2.0)
    
    def default_correlation_func(d,L,sigma):
        C = sigma**2.0 * np.exp((-1*d)/L)
        return C
    
    if cor_func is None:
        used_cor_func = default_correlation_func
    
    x = input_data['data_array'][:,2]
    y = input_data['data_array'][:,3]
    
    # Initialize Covariance Matrix (Dense)
    cm = np.zeros([input_data['ndim'],input_data['ndim']])
    
    rowN = 0 # Tracks spatial index of the row
    # For each row
    for r in range(0,input_data["ndim"]):
        rowM = math.floor(r/input_data["N"])
        rowX = x[rowN]
        rowY = y[rowN]
        for m in range(0,input_data["M"]):
            if m != rowM:
                continue
            for n in range(0,input_data["N"]):
                c = m*input_data["N"] + n
                thisX = x[n]
                thisY = y[n]
                d = math.sqrt((thisX-rowX)**2.0 + (thisY-rowY)**2.0)
                cm[r,c] = used_cor_func(d,L,sigma)
        if rowN == input_data["N"]-1:
            rowN = 0;
        else:
            rowN = rowN+1
    return cm
    
    
    
    
    
    
    
    
    
    
    
    
    
