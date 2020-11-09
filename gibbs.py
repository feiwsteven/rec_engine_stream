#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:22:35 2020

@author: ying
"""

import numpy as np
import pandas as pd
import sklearn 
import scipy
import matplotlib.pyplot as plt

# specify prior parameters
a_r, b_r = 1, 1 # R_{ij} ~ Bern(p), p ~ Beta(a_r, b_r)
alp_u, beta_u = 1, 1 # U_i ~ N(0, sig_u^2), sig_u^2 ~ IG(alp_u, beta_u)
alp_v, beta_v = 1, 1 # V-j ~ N(0, sig_v^2), sig_v^2 ~ IG(alp_v, beta_v)
sig_y2 = 10, # alpha_y ~ N(0,sigma_y^2 I_K), beta_y ~ N(0,sigma_y^2 I_J)

# dimensions
J = 10 # dimension of beta_y and W_i
K = 10 # dimension of alpha_y and X_j
L = 10 # dimension of U_i and V_j 
n = 1000 # nrow(Y)
m = 2000 # ncol(Y)

# generate observed
X = np.random.normal(0,1, (m,K))
W = np.random.normal(0,1, (n,J))
beta0 = np.random.normal(0,1, (J,1)) # true beta0
alpha0 = np.random.normal(0,1, (K,1)) # true alpha0
U = np.random.normal(0,1, (n, L)) # truth: sigma_u = 1
V = np.random.normal(0,1, (m, L)) # truth: sigma_v = 1
Zmeans = np.matmul(W, beta0) + np.matmul(X, alpha0).transpose() + np.matmul(U, V.transpose())
Z = Zmeans + np.random.normal(0,1, (n,m))
R = np.random.binomial(1, 0.1, (n,m)) # truth: p_r = 0.1
Y = 1.0 * (Z>0)
for i in range(n):
    for j in range(m):
        if R[i,j]==0:
            Y[i,j] = float("nan")


# sample from prior to start
alpha_y = np.random.normal(0, np.sqrt(sig_y2), size = (1,K)) # alpha_y ~ N(0,sigma_y^2 I_K)
beta_y = np.random.normal(0, np.sqrt(sig_y2), size = (1,J)) # beta_y ~ N(0,sigma_y^2 I_J)
sigma_u2 = scipy.stats.invgamma.rvs(a= alp_u, scale = beta_u, size=1) # sigma_u^2 ~ IG(alp_u, beta_u)
sigma_v2 = scipy.stats.invgamma.rvs(a= alp_v, scale = beta_v, size=1) # sigma_v^2 ~ IG(alp_v, beta_v)
p_r = np.random.beta(a_r, b_r, size=1)


# sample posterior in gibbs sampler
a_p = a_r + sum(sum(R))
b_p = b_r + sum(sum(1-R))
p_e = np.random.beta(a_p, b_p, size=1)

N = 2000 

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
    print(nn)
    # sample sigma_u2 and sigma_v2
    sigma_u2 = scipy.stats.invgamma.rvs(a= alp_u + n*L/2, scale = beta_u + sum(sum(U*U))/2, size=1)
    sigma_v2 = scipy.stats.invgamma.rvs(a= alp_v + n*L/2, scale = beta_v + sum(sum(V*V))/2, size=1)
    SIG_U2.append(sigma_u2)
    SIG_V2.append(sigma_v2)
    
    # sample alpha_y and beta_y
    sig_alp = np.eye(K)/sig_y2
    sum_r = sum(R)
    for j in range(m):
        sig_alp = sig_alp + sum_r[j] * np.matmul(np.matrix(X[j,]).transpose(), np.matrix(X[j,]))
    sig_alp = np.linalg.inv(sig_alp)
    
    sig_beta = np.eye(J)/sig_y2
    sum_rt = sum(R.transpose())
    for i in range(n):
        sig_beta = sig_beta + sum_rt[i] * np.matmul(np.matrix(W[i,]).transpose(), np.matrix(W[i,]))
    sig_beta = np.linalg.inv(sig_beta)
    
    mu_alp = 0
    sum_rx = sum(R * (Z - np.matmul(W, beta_y.transpose()) - np.matmul(U,V.transpose())))
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


    
    
###############################
# check posterior computation

# check 
# sample from prior N(0,sigma_y^2)

alpha_y = np.random.normal(0, np.sqrt(sig_y2), size = (K,1)) # alpha_y ~ N(0,sigma_y^2 I_K)
beta_y = np.random.normal(0, np.sqrt(sig_y2), size = (J,1)) # beta_y ~ N(0,sigma_y^2 I_J)
    

    
    
    
    
    
    
    
    
    



