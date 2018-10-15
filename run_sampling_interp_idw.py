# This script was for testing an approach to solving the Goren et al. (2014) river
# profile inversion by downsampling river data to a coarser grid using a bilinear
# interpolation.
#

import numpy as np
import pystan
import stan_utility
from grid_utils import create_grid
import os
import sampling_utils
import pandas
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ## ------------------------RUN PARAMETERS ------------------------------##
    out_dir_name = "Samples_IDW_Testing_02232018"
    input_file = "./inputs/presorted_drainage_data_example.txt"
    model_src = "./models/multinormal_with_idw_interp.txt"
    model_compiled = "./models/multinormal_with_idw_interp.pkl"
    compile_model = True # Flag for whether to recompile the Stan model
    
    # --- These variables are named to match their counterparts in the Stan model
    M = 5
    nx = 10
    ny = 10
    sdevData = 5
    sdevModel = 1
    minModel = 0
    maxModel = 20
    meanModel = 0
    
    # MCMC Parameters
    n_chains = 4        # Number of MCMC Chains
    n_iter = 500        # Number of MCMC iterations
    n_warmup = None     # Number of warmup iterations
    n_jobs = 1          # Number of threads to use (multithreading is wonky on Windows)

    ## ---------------------------------------------------------------------##
    
    np.random.seed(42)
    
    # Prepare Output Directory
    out_dir = os.path.join(os.getcwd(),out_dir_name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Load Dataset
    print("--> Preparing Dataset...")
    input_data = sampling_utils.prepare_model_data(input_file,M,sdevData,
                                                   sdevModel,meanModel,minModel,
                                                   maxModel)
    N = input_data["N"]
    ndim = N*M
    print("Done.")

    # Create uplift grid
    print("--> Gridding model...")
    x = input_data['data_array'][:,2]
    y = input_data['data_array'][:,3]
    grid = create_grid(nx,ny,x,y,N,M=M)
    print("Done.")
    
    # Prepare STAN Model
    print("--> Preparing STAN Model...")
    if compile_model:
        print("Compiling (Verbose)...")
        sm,sm_pkl = sampling_utils.create_model(model_src)
    else:
        print("Loading Precompiled...")
        sm = sampling_utils.load_model(model_compiled)
    print("Done.")
    
    # Prepare Dataset to be passed to STAN
    sampler_data = {
            'N':input_data['N'],
            'M':input_data['ndim'],
            'uMin':minModel,
            'uMean':meanModel,
            'uSdev':sdevModel,
            'sdevData':sdevData,
            'A':input_data['A'],
            'y':input_data['d0'].copy(),
            'nSteps':M,
            'nNodes':grid["n_nodes"],
            'weights_idw':grid["idw_weights"],
            }

    # Run MCMC Sampler
    def generate_initial_values(dim):
        
        return {
                'u':np.random.uniform(low=minModel,high=maxModel,size=[dim]),
                'L_Omega':np.random.uniform(low=-0.1,high=0.1,size=[dim,dim]),
                'L_sigma':np.random.uniform(low=minModel,high=sdevModel,size=[dim])
                }
    init_list = []
    for i_chain in range(0,n_chains):
        init_list.append(generate_initial_values(grid['n_nodes']*M))
    print("--> Sampling...")
    control= {"adapt_delta":0.8
               }
    samplesFile = os.path.join(out_dir,'samples_M{0}.dat'.format(M))
    fit = sm.sampling(data=sampler_data,iter=n_iter,warmup=n_warmup,n_jobs=n_jobs,seed=42,
                      chains=n_chains,sample_file=samplesFile,
                      init=init_list,verbose=True,control=control)

    print("Done.")
    
    # Check Diagnostics on Output Samples
    print("--> Running Diagnostics on Model Fit...")
    stan_utility.check_all_diagnostics(fit)
    
    # Gather Mean Result and its SDEV
    N = grid["n_nodes"]
    
    summary = fit.summary()['summary']
    mean = summary[:,0][0:N*M]
    sd = summary[:,2][0:N*M]

    # Transfer results to a N*M matrix
    mean_array = np.reshape(mean,[N,M])
    mean_df = pandas.DataFrame(mean_array)
    mean_df['n'] = np.arange(1,N+1)
    mean_df['p'] = [input_data["row_to_global"][x] for x in range(0,N)]
    np.savetxt(os.path.join(out_dir,'mean.csv'),mean_df.values)
    sd_array = np.reshape(sd,[N,M])
    sd_df = pandas.DataFrame(sd_array)
    sd_df['n'] = np.arange(1,N+1)
    sd_df['p'] = [input_data["row_to_global"][x] for x in range(0,N)]
    np.savetxt(os.path.join(out_dir,'sd.csv'),sd_df.values)
    
    # Run forward calc for Mean Estimate and Compute RMS Error
    
    def interpGridToStream(grid,mean,M,N,n_nodes):
        u_stream = np.zeros([N*M])
        for i in range(0,M):
            u_stream[i*N:i*N+N] = np.dot(grid["idw_weights"],mean[i*n_nodes:i*n_nodes+n_nodes])
        return u_stream
        
    n_spatial = input_data["N"]
    mean_interp = interpGridToStream(grid,mean,M,n_spatial,N)
    sd_interp = interpGridToStream(grid,sd,M,n_spatial,N)
    mean_interp_array = np.reshape(mean_interp,[n_spatial,M])
    sd_interp_array = np.reshape(sd_interp,[n_spatial,M])
    mean_fwd = input_data["A"].dot(mean_interp)
    rmse = np.sqrt(np.sum(np.square(mean_fwd - input_data["d0"])) / n_spatial)
    z_fwd = mean_fwd + input_data["minElev"]
    z_fwd = z_fwd.transpose()
    
    print('Mean Posterior rmse: '  + str(rmse))
    print("DONE!")

    fig = plt.figure(1)
    fig.clf()
    for i in range(0,M*2):
        m_i = int(np.floor(i/2))
        plt.subplot(M,2,i+1)
        profileInds =  [x for x in range(0,n_spatial)]
        u_plot_max = mean_interp.max() + sd_interp.max()
        u_plot_min = 0
        # Right column
        if (i % 2) != 0:
            plt.plot(profileInds,mean_interp_array[:,m_i],'k-')
            plt.fill_between(profileInds,mean_interp_array[:,m_i]-sd_interp_array[:,m_i],
                             mean_interp_array[:,m_i]+sd_interp_array[:,m_i],alpha=0.75)
            plt.ylim(u_plot_min,u_plot_max)
            plt.ylabel('u*')
        else:   
            line_mean = plt.plot(profileInds,z_fwd,label='Mean Posterior Model')
            line_data = plt.plot(profileInds,input_data['d0']+input_data['minElev'],label='Observed')
            if (i==0):
                plt.legend(prop={'size':14})
            plt.ylabel('z [m]')
                
    fig.suptitle("Model Results for SIGMA_DATA={0}, M={1}".format(sdevData,M))
    thisPlotFile = os.path.join(out_dir,"PLOT_M-"+str(M)+".png")
    plt.savefig(thisPlotFile)
    plt.show()

    print(fit)
