#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:22:35 2020

@author: ying
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import yule_walker
import multiprocessing

dir_name = "/home/yingjin/gibbs/rec_engine_stream/simulations/gibbs/workflow"
os.chdir(dir_name)
from gibbs_funcs import base_gibbs
from gibbs_funcs import base_gibbs_wrapper
from gibbs_funcs import initial_gibbs
from gibbs_funcs import EffectiveSize


### simulations
    
# specify prior parameters
a_r, b_r = 1, 1 # R_{ij} ~ Bern(p), p ~ Beta(a_r, b_r)
alp_u, beta_u = 1, 1 # U_i ~ N(0, sig_u^2), sig_u^2 ~ IG(alp_u, beta_u)
alp_v, beta_v = 1, 1 # V-j ~ N(0, sig_v^2), sig_v^2 ~ IG(alp_v, beta_v)
sig_y2 = 10 # alpha_y ~ N(0,sigma_y^2 I_K), beta_y ~ N(0,sigma_y^2 I_J)

coef_scale = float(os.environ["arg1"])
mnid = int(os.environ["arg2"])
random_dim = int(os.environ["arg3"])

ns = [100, 1000, 5000]
ms = [20, 200, 2000]
n = ns[mnid]
m = ms[mnid]

# dimensions
J = random_dim # dimension of beta_y and W_i
K = random_dim # dimension of alpha_y and X_j
L = random_dim # dimension of U_i and V_j 
#n = 1000 # nrow(Y)
#m = 2000 # ncol(Y)

# generate observed
X = np.random.normal(0,1, (m,K))
W = np.random.normal(0,1, (n,J))
beta0 = (2*np.random.binomial(1,0.5,J).reshape((J,1)) -1)* coef_scale #np.random.normal(0,1, (J,1)) # true beta0 ~ uniform, larger value
alpha0 = (2*np.random.binomial(1,0.5,J).reshape((J,1)) -1)* coef_scale #np.random.normal(0,1, (K,1)) # true alpha0
U = np.random.normal(0,1, (n, L)) # truth: sigma_u = 1
V = np.random.normal(0,1, (m, L)) # truth: sigma_v = 1
Zmeans = np.matmul(W, beta0) + np.matmul(X, alpha0).transpose() + np.matmul(U, V.transpose())
Z = Zmeans + np.random.normal(0,1, (n,m)) # var of Zmeans should be large
R = np.random.binomial(1, 0.01, (n,m)) # truth: p_r = 0.1
Y = 1.0 * (Z>0)
for i in range(n):
    for j in range(m):
        if R[i,j]==0:
            Y[i,j] = float("nan")

snr = Zmeans[R>0].std()**2/ Z[R>0].std()**2

'''
# sample from prior to start ~ start from estimator
alpha_y = np.random.normal(0, np.sqrt(sig_y2), size = (1,K)) # alpha_y ~ N(0,sigma_y^2 I_K)
beta_y = np.random.normal(0, np.sqrt(sig_y2), size = (1,J)) # beta_y ~ N(0,sigma_y^2 I_J)
sigma_u2 = stats.invgamma.rvs(a= alp_u, scale = beta_u, size=1) # sigma_u^2 ~ IG(alp_u, beta_u)
sigma_v2 = stats.invgamma.rvs(a= alp_v, scale = beta_v, size=1) # sigma_v^2 ~ IG(alp_v, beta_v)
p_r = np.random.beta(a_r, b_r, size=1)


# sample initial from kinds of posterior / expectation
U = np.random.normal(0,1, (n, L)) # sample iid N(0,1)
V = np.random.normal(0,1, (m, L)) # sample iid N(0,1)
Z = np.zeros((n,m))
for i in range(n):
    for j in range(m):
        if R[i,j] == 1 and Y[i,j] == 0:
            Z[i,j] = -np.sqrt(2/np.pi)
        elif R[i,j] ==1 and Y[i,j] == 1:
            Z[i,j] = np.sqrt(2/np.pi)

# sample posterior in gibbs sampler
a_p = a_r + sum(sum(R))
b_p = b_r + sum(sum(1-R))
p_e = np.random.beta(a_p, b_p, size=1)
'''

N_tot = 50000
N_sing = 500 # num of iterations before each check
n_chain = 3

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=n_chain)
    
    # initiate
    args = []
    for i in range(n_chain):
        # initial_gibbs returns tuple (Z,U,V,alpha_y,beta_y,sigma_u2,sigma_v2)
        init = initial_gibbs(Y,W,X,Z,R,L,sig_y2, alp_u, beta_u, alp_v, beta_v, a_r, b_r)
        new_arg = list((N_sing,R,Y,W,X) + init + (alp_u, beta_u, alp_v, beta_v, sig_y2))
        args.append(new_arg)
    
    count = 0
    
    # record results
    all_alpy = [pd.DataFrame()] * n_chain
    all_betay = [pd.DataFrame()] * n_chain
    all_sigu2 = [pd.DataFrame()] * n_chain
    all_sigv2 = [pd.DataFrame()] * n_chain
    all_paras = [pd.DataFrame()] * n_chain
    
    #########################
    # burnin
    if_continue = True
    while count < N_tot and if_continue:
        
        # run a batch of N_sing iterations
        results = pool.map(base_gibbs_wrapper, args)
        count += N_sing
        
        # process the results
        for i in range(n_chain):
            # subtract results for each chain
            tmp_res = results[i]
            tmp_alpy = tmp_res[2]
            tmp_betay = tmp_res[3]
            tmp_sigu2 = tmp_res[0]
            tmp_sigv2 = tmp_res[1]
            UU = tmp_res[4]
            VV = tmp_res[5]
            ZZ = tmp_res[6]
            
            # update args
            new_arg  = list((N_sing, R,Y,W,X, ZZ[-1], UU[-1], VV[-1],tmp_alpy[-1], tmp_betay[-1], 
                             tmp_sigu2[-1], tmp_sigv2[-1], alp_u, beta_u, alp_v, beta_v, sig_y2))
            args[i] = new_arg
            
            # concatenate results
            tmp_alpy_mat = np.matrix(np.zeros(shape=(N_sing,J)))
            tmp_betay_mat = np.matrix(np.zeros(shape=(N_sing,J)))
            
            for jj in range(N_sing):
                tmp_alpy_mat[jj,] = tmp_alpy[jj]
                tmp_betay_mat[jj,] = tmp_betay[jj]
            
            all_alpy[i] = pd.concat([all_alpy[i], pd.DataFrame(tmp_alpy_mat)]) 
            all_betay[i] = pd.concat([all_betay[i], pd.DataFrame(tmp_betay_mat)]) 
            all_sigu2[i] = pd.concat([all_sigu2[i], pd.DataFrame(tmp_sigu2)]) 
            all_sigv2[i] = pd.concat([all_sigv2[i], pd.DataFrame(tmp_sigu2)]) 
            
            all_paras[i] = pd.concat([all_paras[i],
                                      pd.concat([pd.DataFrame(tmp_alpy_mat), 
                                                 pd.DataFrame(tmp_betay_mat), 
                                                 pd.DataFrame(tmp_sigu2), 
                                                 pd.DataFrame(tmp_sigv2)],axis=1)])
        
        
        # diagnostic for burnin
        # Gelman-Rubin test
        if_pass = True
        n_paras = (all_paras[0]).shape[1]
        NN = int(all_paras[0].shape[0]/2)
        
        hatrs = np.array([0.0]*n_paras)
        for jj in range(n_paras):
            means = []
            stds = []
            for ii in range(n_chain):
                mean = all_paras[ii].iloc[NN:NN*2,jj].mean()
                sd = all_paras[ii].iloc[NN:NN*2,jj].std()
                means.append(mean)
                stds.append(sd)
            means = np.array(means)
            stds = np.array(stds)
            ww = np.mean(stds**2)
            bb = np.std(means) **2 * NN
            
            hatvar = (1-1/NN)*ww + bb/NN
            hatr = np.sqrt(hatvar/ww)
            hatrs[jj] = hatr
        
        if_pass = sum((1.0<=hatrs)&(hatrs<=1.1)) == n_paras
        if if_pass:
            if_continue = False
            
        
    #########################
    # after burn-in procedure
    # start over with the latest updated arguments
    after_alpy = [pd.DataFrame()] * n_chain
    after_betay = [pd.DataFrame()] * n_chain
    after_sigu2 = [pd.DataFrame()] * n_chain
    after_sigv2 = [pd.DataFrame()] * n_chain
    after_paras = [pd.DataFrame()] * n_chain
    after_times = [pd.DataFrame()] * n_chain # record time of each run
    after_loglikes = [pd.DataFrame()] * n_chain
    
    count = 0
    if_continue = True
    while count < N_tot and if_continue:
        
        # run a batch of N_sing iterations
        results = pool.map(base_gibbs_wrapper, args)
        count += N_sing
        
        # process and record the results
        for i in range(n_chain):
            # subtract results for each chain
            tmp_res = results[i]
            tmp_alpy = tmp_res[2]
            tmp_betay = tmp_res[3]
            tmp_sigu2 = tmp_res[0]
            tmp_sigv2 = tmp_res[1]
            UU = tmp_res[4]
            VV = tmp_res[5]
            ZZ = tmp_res[6]
            tmp_loglikes = tmp_res[7] # log likelihood for each sampling
            tmp_times = tmp_res[8] # running time of every 100 iterations
            
            # update args
            new_arg  = list((N_sing, R,Y,W,X, ZZ[-1], UU[-1], VV[-1],tmp_alpy[-1], tmp_betay[-1], 
                             tmp_sigu2[-1], tmp_sigv2[-1], alp_u, beta_u, alp_v, beta_v, sig_y2))
            args[i] = new_arg
            
            # concatenate results
            tmp_alpy_mat = np.matrix(np.zeros(shape=(N_sing,J)))
            tmp_betay_mat = np.matrix(np.zeros(shape=(N_sing,J)))
            
            for jj in range(N_sing):
                tmp_alpy_mat[jj,] = tmp_alpy[jj]
                tmp_betay_mat[jj,] = tmp_betay[jj]
            
            # record the results
            after_alpy[i] = pd.concat([after_alpy[i], pd.DataFrame(tmp_alpy_mat)]) 
            after_betay[i] = pd.concat([after_betay[i], pd.DataFrame(tmp_betay_mat)]) 
            after_sigu2[i] = pd.concat([after_sigu2[i], pd.DataFrame(tmp_sigu2)]) 
            after_sigv2[i] = pd.concat([after_sigv2[i], pd.DataFrame(tmp_sigv2)]) 
            after_times[i] = pd.concat([after_times[i], pd.DataFrame(tmp_times)])
            after_loglikes[i] = pd.concat([after_loglikes[i], pd.DataFrame(tmp_loglikes)])
            
            # concatenate into one dataframe for analysis
            tmp_all = pd.concat([pd.DataFrame(tmp_alpy_mat), pd.DataFrame(tmp_betay_mat)],axis=1) 
            tmp_all['sigu'] = tmp_sigu2
            tmp_all['sigv'] = tmp_sigv2
            if after_paras[i].shape[0] >0:
                after_paras[i].columns = tmp_all.columns
            
            after_paras[i] = pd.concat([after_paras[i],tmp_all])
        
        # calculate efficient sample size
        ESS_list = []
        if_stop = True
        for i in range(n_chain):
            ESS = EffectiveSize(after_paras[i])
            ESS_list.append(ESS)
            i_continue = sum(np.array(ESS)<500) > 0
            if i_continue:
                if_stop = False
            
        if if_stop:
            if_continue = False
        
        
    #########################
    # after stopping
    # put results together
    all_parameters = pd.DataFrame()
    for i in range(n_chain):
        all_parameters = pd.concat([all_parameters, after_paras[i]])
    df_alpy = all_parameters.iloc[:,0:L]
    df_betay = all_parameters.iloc[:,L:(2*L)]
    df_sigu = all_parameters.iloc[:,2*L]
    df_sigv = all_parameters.iloc[:,2*L+1]
    
    df_loglike = pd.DataFrame()
    df_times = pd.DataFrame()
    for i in range(n_chain):
        df_loglike = pd.concat([df_loglike, after_loglikes[i]])
        df_times = pd.concat([df_times, after_times[i]])
    df_snr = pd.DataFrame([snr])
        
            
    '''
    #res = gibbs(N,R,Y,W,X,Z,U,V,alpha_y,beta_y,sigma_u2,sigma_v2,alp_u,beta_u,alp_v,beta_v,sig_y2)
    
    
    # wrap up results
    conv_itrs = int(res[8])
    
    df_sigu = pd.DataFrame(res[0])
    df_sigv = pd.DataFrame(res[1])
    mt_alpy = np.matrix(np.zeros(shape=(conv_itrs+1,J)))
    mt_betay = np.matrix(np.zeros(shape=(conv_itrs+1,J)))
    # append truth
    mt_alpy[0,] = alpha0.reshape(-1)
    mt_betay[0,] = beta0.reshape(-1)
    
    for nn in range(conv_itrs):
        mt_alpy[nn+1,] = res[2][nn]
        mt_betay[nn+1,] = res[3][nn]
    df_alpy = pd.DataFrame(mt_alpy)
    df_betay = pd.DataFrame(mt_betay)
    df_times = pd.DataFrame(res[7])
    '''

    # MSE of posterior mean
    sigu_err = (df_sigu - 1).mean()[0] #np.sqrt(np.mean(np.power(df_sigu-1,2)))
    sigv_err = (df_sigv - 1).mean()[0]
    alpy_err = np.sqrt(np.power(df_alpy - alpha0.reshape(-1).mean(),2).mean())
    betay_err = np.sqrt(np.power(df_betay - beta0.reshape(-1).mean(),2).mean())
    df_res = pd.DataFrame([coef_scale,n,m,L,sigu_err,sigv_err,alpy_err,betay_err])
    df_res.to_csv(dir_name+"/results/res_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
    df_times.to_csv(dir_name+"/results/time_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
    
    df_snr.to_csv(dir_name+"/results/snr_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
    df_sigu.to_csv(dir_name+"/results/sigu_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
    df_sigv.to_csv(dir_name+"/results/sigv_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
    df_alpy.to_csv(dir_name+"/results/alpy_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
    df_betay.to_csv(dir_name+"/results/betay_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")  
    
    
    
    



