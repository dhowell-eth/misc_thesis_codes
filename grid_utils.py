    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 11:11:33 2018

@author: dorran
"""

import numpy as np
import math

def create_grid(nx,ny,x,y,N,M=None,p=1):

    # Create grids
    grid_x_bounds = [x.min() - 10, x.max() + 10]
    grid_y_bounds = [y.min() - 10, y.max() + 10]
    
    x_size = grid_x_bounds[1] - grid_x_bounds[0]
    y_size = grid_y_bounds[1] - grid_y_bounds[0]
    dx = x_size/(nx-1)
    dy = y_size/(ny-1)
    
    x_grid = np.arange(x.min(),grid_x_bounds[1]+dx,dx)
    y_grid = np.arange(y.min(),grid_y_bounds[1]+dy,dy)
    X,Y = np.meshgrid(x_grid,y_grid,sparse=False, indexing='ij')
    ids = np.zeros_like(X)
    for j in range(0,ny):
        for i in range(0,nx):
            this_id = (j*ny + i)
            ids[i,j] = this_id

    # Assign weights to each stream point
    weights2 =  np.zeros([x.size,4])
    corner_ids = np.zeros(4*N,dtype=int)
    
    for s_i in range(0,x.size):
        
        # Get indices for Bottom Left Grid Node
        #  (i+1,j)   (i+1,j+1)
        #  2---------3
        #  |         |
        #  |         |
        #  |      x  |
        #  |         |
        #  1---------4
        #  (i,j)    (i,j+1)
        
        i = int(np.fix( (x[s_i] - x_grid[0]) / dx ))
        j = int(np.fix( (y[s_i] - y_grid[0]) / dy ))

        distanceX = x[s_i] - X[i,j]
        distanceY = y[s_i] - Y[i,j]
    
        # Determine weights
        WT1 = (1-distanceX/dx) * (1-distanceY/dy)
        WT2 = (1-distanceX/dx) * (distanceY/dy)
        WT3 = (distanceX/dx) * (distanceY/dy)
        WT4 = (distanceX/dx) * (1-distanceY/dy)
        
        corner_ids[4*s_i:4*s_i+4] = [ids[i,j],ids[i,j+1],ids[i+1,j+1],ids[i+1,j]]
    
        if distanceX > dx or distanceY>dy:
            print("s_i:"+str(s_i))
            print("i,j: {0},{1}".format(i,j))
            raise Exception('DISTANCE GREATER THAN NODE SPACING!')
            
        weights2[s_i,:] = [WT1,WT2,WT3,WT4]
    
    
    # Trim grid nodes to those containing 1+ stream nodes
    ids_flat = ids.flatten('F').tolist()
    missing_nodes = [i for i in ids_flat if i not in corner_ids]
    uplift_indices = corner_ids.copy()
    subtract = 0
    for i,node_id in enumerate(missing_nodes):
        ids_flat.remove(node_id)
        greater_inds = uplift_indices+subtract > node_id
        uplift_indices[greater_inds] = uplift_indices[greater_inds] - 1
        subtract = subtract + 1
    uplift_indices = uplift_indices.reshape([4,x.size],order='F')
    uplift_indices = uplift_indices.transpose()

    # Get x,y for trimmed nodes
    xU = []
    yU = []
    for node_id in ids_flat:
        i,j = np.where(ids == node_id)
        xU.append(X[j,i][0])
        yU.append(Y[j,i][0])
        
    if M is not None:
        sections_weights = []
        sections_indices = []
        for i in range(0,M):
            sections_weights.append(weights2.copy())
            sections_indices.append(uplift_indices.copy())
        weights2 = np.vstack(sections_weights)
        uplift_indices = np.vstack(sections_indices)

    # Compute IDW terms
    idw_weights = np.zeros([x.size,len(ids_flat)])    
    # For each stream point
    for s_i in range(0,x.size):
        # Loop through trimmed grid nodes
        distances = np.zeros([len(ids_flat)])
        for n_i,node_id in enumerate(ids_flat):
            i,j = np.where(ids == node_id)
            distance = math.sqrt((x[s_i]-X[i,j])**2.0 + (y[s_i]-Y[i,j])**2.0)
            if distance == 0:
                distances[:] = 0.0
                distances[n_i] = 1
            else:
                distances[n_i] = 1 / (distance**p)
    
        # Assign terms to output arrays
        idw_weights[s_i,:] = distances / distances.sum()
    # Now to compute idw values at stream points:
    # dot(idw_weights,u_grid_nodes)
    # e.g. u_idw_stream = idw_weights.dot(exampleUplift)
            
    # Prepare outputs
    outputs = {"X":X,
               "Y":Y,
               "xU":xU,
               "yU":yU,
               "ids":ids,
               "weights":weights2,
               "idw_weights":idw_weights,
               "corner_indices":uplift_indices,
               "n_nodes":len(ids_flat),
               "trimmed_ids":ids_flat}
    
    return outputs



