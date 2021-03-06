#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:22:35 2020

@author: ying
"""

import os
import numpy as np
import pandas as pd
import sklearn 
import scipy
import matplotlib.pyplot as plt


#### the basic gibbs sampler
# N is the number of iterations
# R is the indicator matrix of observations, dim(R)=(n,m)
# Y is the ratings (binary), dim(Y)=(n,m)
# W is the user feature matrix, dim(W)=(n,J)
# X is the item feature matrix, dim(X)=(m,K)
# U is the initialized noise matrix, dim(U)=(n,L)
# V is the initialized noise matrix, dim(V)=(m,L)
# L is the dimension of U_i and V_j
# alpha_y, beta_y, sigma_u2, sigma_v2 are initial values from prior
# alp_u, beta_u, alp_v, beta_v, sig_y2 are prior parameters (we choose sig_y2 large for a flat prior)
def gibbs(N, R, Y, X, U, V, alpha_y, beta_y, sigma_u2, sigma_v2, alp_u, beta_u, alp_v, beta_v, sig_y2):
    n, m = R.shape
    J = W.shape[1]
    K = X.shape[1]
    L = U.shape[1]
    
    SIG_U2 = []
    SIG_V2 = []
    ALP_Y = []
    BETA_Y = []
    UU = []
    VV = []
    ZZ = []
    
    OBi = []
    for i in range(n):
        observed = [j for j in range(m) if R[i,j]==1.0]
        OBi.append(observed)
    OBj = []
    for j in range(m):
        observed = [i for i in range(n) if R[i,j]==1.0]
        OBj.append(observed)
    
    for nn in range(N):
        #print(nn)
        # sample sigma_u2 and sigma_v2
        sigma_u2 = scipy.stats.invgamma.rvs(a= alp_u + n*L/2, scale = beta_u + sum(sum(U*U))/2, size=1)
        sigma_v2 = scipy.stats.invgamma.rvs(a= alp_v + m*L/2, scale = beta_v + sum(sum(V*V))/2, size=1)
        SIG_U2.append(sigma_u2[0])
        SIG_V2.append(sigma_v2[0])
        
        # sample alpha_y and beta_y
        sig_alp = np.eye(K)/sig_y2
        sum_r = sum(R)
        for j in range(m):
            sig_alp = sig_alp + sum_r[j] * np.matmul(np.matrix(X[j,]).transpose(), np.matrix(X[j,]))
        sig_alp = np.linalg.inv(sig_alp) #choleskey decomp
        
        sig_beta = np.eye(J)/sig_y2
        sum_rt = sum(R.transpose())
        for i in range(n):
            sig_beta = sig_beta + sum_rt[i] * np.matmul(np.matrix(W[i,]).transpose(), np.matrix(W[i,]))
        sig_beta = np.linalg.inv(sig_beta)
        
        mu_alp = 0
        sum_rx = sum(R * (Z - np.matmul(W, beta_y.transpose()) - np.matmul(U,V.transpose())))#save U'V
        for j in range(m):
            mu_alp = mu_alp + sum_rx[j] * X[j,]
        mu_alp = np.matmul(sig_alp, mu_alp)
        
        mu_beta = 0
        sum_rw = sum(R.transpose() * (Z.transpose() - np.matmul(X, alpha_y.transpose()) - np.matmul(V,U.transpose())))
        for i in range(n):
            mu_beta = mu_beta + sum_rw[i] * W[i,]
        mu_beta = np.matmul(sig_beta, mu_beta)
        
        alpha_y = np.random.multivariate_normal(mean = np.asarray(mu_alp).reshape(-1), cov=sig_alp, size=1)
        beta_y = np.random.multivariate_normal(mean = np.asarray(mu_beta).reshape(-1), cov=sig_beta, size=1)
        ALP_Y.append(alpha_y)
        BETA_Y.append(beta_y)
        
        # sample U_i and V_j
        for i in range(n):
            sig_ui = np.eye(L) / sigma_u2
            mu_ui = 0
            for j in OBi[i]:
                sig_ui = sig_ui + R[i,j] * np.matmul(np.matrix(V[j,]).transpose(), np.matrix(V[j,]))
                mu_ui = mu_ui + R[i,j] * (Z[i,j] - sum(sum(W[i,]*beta_y)) - sum(sum(X[j,]*alpha_y))) * V[j,]
            sig_ui = np.linalg.inv(sig_ui)
            mu_ui = np.matmul(sig_ui, mu_ui.reshape(L,1))
            U[i,] = np.random.multivariate_normal(mean = np.asarray(mu_ui).reshape(-1), cov=sig_ui, size=1)
        
        for j in range(m):
            sig_vj = np.eye(L) / sigma_v2
            mu_vj = 0
            for i in OBj[j]:
                sig_vj = sig_vj + R[i,j] * np.matmul(np.matrix(U[i,]).transpose(), np.matrix(U[i,]))
                mu_vj = mu_vj + R[i,j] * (Z[i,j] - sum(sum(W[i,]*beta_y)) - sum(sum(X[j,]*alpha_y))) * U[i,]
            sig_vj = np.linalg.inv(sig_vj)
            mu_vj = np.matmul(sig_vj, mu_vj.reshape(L,1))
            V[j,] = np.random.multivariate_normal(mean = np.asarray(mu_vj).reshape(-1), cov=sig_vj, size=1)
        UU.append(U)
        VV.append(V)
            
        # sample Z_ij
        zmean = np.matmul(W, beta_y.reshape((J,1))) + np.matmul(X, alpha_y.reshape((K,1))).transpose() + np.matmul(U, V.transpose())
        for i in range(n):
            for j in OBi[i]:
                if Y[i,j] == 1:
                    Z[i,j] = scipy.stats.truncnorm.rvs(-zmean[i,j], np.inf, loc=zmean[i,j],scale=1,size=1)
                else:
                    Z[i,j] = scipy.stats.truncnorm.rvs(-np.inf, -zmean[i,j], loc=zmean[i,j],scale=1,size=1)
        ZZ.append(Z)
    
    return SIG_U2, SIG_V2, ALP_Y, BETA_Y, UU, VV, ZZ

    
# specify prior parameters
a_r, b_r = 1, 1 # R_{ij} ~ Bern(p), p ~ Beta(a_r, b_r)
alp_u, beta_u = 1, 1 # U_i ~ N(0, sig_u^2), sig_u^2 ~ IG(alp_u, beta_u)
alp_v, beta_v = 1, 1 # V-j ~ N(0, sig_v^2), sig_v^2 ~ IG(alp_v, beta_v)
sig_y2 = 10, # alpha_y ~ N(0,sigma_y^2 I_K), beta_y ~ N(0,sigma_y^2 I_J)

coef_scale = os.environ["arg1"]
n = os.environ["arg2"]
m = os.environ["arg3"]
random_dim = os.environ["arg4"]

# dimensions
J = random_dim # dimension of beta_y and W_i
K = random_dim # dimension of alpha_y and X_j
L = random_dim # dimension of U_i and V_j 
#n = 1000 # nrow(Y)
#m = 2000 # ncol(Y)

# generate observed
X = np.random.normal(0,1, (m,K))
W = np.random.normal(0,1, (n,J))
beta0 = (2*np.random.binomial(1,0.5,J).reshape((J,1))-1)* coef_scale #np.random.normal(0,1, (J,1)) # true beta0 ~ uniform, larger value
alpha0 = (2*np.random.binomial(1,0.5,J).reshape((J,1))-1)* coef_scale #np.random.normal(0,1, (K,1)) # true alpha0
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
sigma_u2 = scipy.stats.invgamma.rvs(a= alp_u, scale = beta_u, size=1) # sigma_u^2 ~ IG(alp_u, beta_u)
sigma_v2 = scipy.stats.invgamma.rvs(a= alp_v, scale = beta_v, size=1) # sigma_v^2 ~ IG(alp_v, beta_v)
p_r = np.random.beta(a_r, b_r, size=1)


# sample posterior in gibbs sampler
a_p = a_r + sum(sum(R))
b_p = b_r + sum(sum(1-R))
p_e = np.random.beta(a_p, b_p, size=1)

N = 1000



res = gibbs(N,R,Y,X,U,V,alpha_y,beta_y,sigma_u2,sigma_v2,alp_u,beta_u,alp_v,beta_v,sig_y2)

# wrap up results
df_snr = pd.DataFrame([snr])
df_sigu = pd.DataFrame(res[0])
df_sigv = pd.DataFrame(res[1])
mt_alpy = np.matrix(np.zeros(shape=(N,J)))
mt_betay = np.matrix(np.zeros(shape=(N,J)))
for n in range(N):
    mt_alpy[n,] = res[2][n]
    mt_betay[n,] = res[3][n]
df_alpy = pd.DataFrame(mt_alpy)
df_betay = pd.DataFrame(mt_betay)

df_sigu.to_csv("sigu_scale_"+str(coef_scale)+"n_"+str(n)+"m_"+str(m)+"L_"+str(L)+".csv")
df_sigv.to_csv("sigv_scale_"+str(coef_scale)+"n_"+str(n)+"m_"+str(m)+"L_"+str(L)+".csv")
df_alpy.to_csv("alpy_scale_"+str(coef_scale)+"n_"+str(n)+"m_"+str(m)+"L_"+str(L)+".csv")
df_betay.to_csv("betay_scale_"+str(coef_scale)+"n_"+str(n)+"m_"+str(m)+"L_"+str(L)+".csv")  
    
    
    
    



