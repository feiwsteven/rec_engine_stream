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

dir_name = "/home/yingjin/gibbs/rec_engine_stream/simulations/gibbs/workflow"
os.chdir(dir_name)
from gibbs_funcs import gibbs

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

# sample from prior to start ~ start from estimator
alpha_y = np.random.normal(0, np.sqrt(sig_y2), size = (1,K)) # alpha_y ~ N(0,sigma_y^2 I_K)
beta_y = np.random.normal(0, np.sqrt(sig_y2), size = (1,J)) # beta_y ~ N(0,sigma_y^2 I_J)
sigma_u2 = stats.invgamma.rvs(a= alp_u, scale = beta_u, size=1) # sigma_u^2 ~ IG(alp_u, beta_u)
sigma_v2 = stats.invgamma.rvs(a= alp_v, scale = beta_v, size=1) # sigma_v^2 ~ IG(alp_v, beta_v)
p_r = np.random.beta(a_r, b_r, size=1)


# sample initial from prior
U = np.random.normal(0,1, (n, L)) # sample iid N(0,1)
V = np.random.normal(0,1, (m, L)) # sample iid N(0,1)
Zmeans = np.matmul(W, beta_y.reshape(J,1)) + np.matmul(X, alpha_y.reshape(J,1)).transpose() + np.matmul(U, V.transpose())
Z = Zmeans

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

N = 50000



res = gibbs(N,R,Y,W,X,Z,U,V,alpha_y,beta_y,sigma_u2,sigma_v2,alp_u,beta_u,alp_v,beta_v,sig_y2)


# wrap up results
conv_itrs = int(res[8])
df_snr = pd.DataFrame([snr])
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


# MSE of posterior mean
sigu_err = (df_sigu - 1).mean()[0] #np.sqrt(np.mean(np.power(df_sigu-1,2)))
sigv_err = (df_sigv - 1).mean()[0]
alpy_err = np.sqrt(np.power(pd.DataFrame(mt_alpy[1:,] - alpha0.reshape(-1)).mean(),2).mean())
betay_err = np.sqrt(np.power(pd.DataFrame(mt_betay[1:,] - beta0.reshape(-1)).mean(),2).mean())
df_res = pd.DataFrame([coef_scale,n,m,L,sigu_err,sigv_err,alpy_err,betay_err,conv_itrs])
df_res.to_csv(dir_name+"/results/res_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
df_times.to_csv(dir_name+"/results/time_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")

df_snr.to_csv(dir_name+"/results/snr_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
df_sigu.to_csv(dir_name+"/results/sigu_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
df_sigv.to_csv(dir_name+"/results/sigv_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
df_alpy.to_csv(dir_name+"/results/alpy_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")
df_betay.to_csv(dir_name+"/results/betay_scale_"+str(coef_scale)+"nm_"+str(mnid)+"L_"+str(L)+".csv")  
    
    
    
    



