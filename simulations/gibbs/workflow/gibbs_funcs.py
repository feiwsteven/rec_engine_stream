#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 20:22:35 2020

@author: ying
"""

import os
import numpy as np
import pandas as pd
import time
from scipy import stats
import scipy
from statsmodels.regression.linear_model import yule_walker
import multiprocessing
from statsmodels.tsa.ar_model import AR

import warnings
warnings.filterwarnings('ignore')

#### gibbs sampler with parallel computing, diagnosis, etc.
# N is the number of iterations
# R is the indicator matrix of observations, dim(R)=(n,m)
# Y is the ratings (binary), dim(Y)=(n,m)
# W is the user feature matrix, dim(W)=(n,J)
# X is the item feature matrix, dim(X)=(m,K)
# Z is the initialized indicators
# U is the initialized noise matrix, dim(U)=(n,L)
# V is the initialized noise matrix, dim(V)=(m,L)
# L is the dimension of U_i and V_j
# alpha_y, beta_y, sigma_u2, sigma_v2 are initial values from prior
# alp_u, beta_u, alp_v, beta_v, sig_y2 are prior parameters (we choose sig_y2 large for a flat prior)
# N_tot is the total number of iterations
# N_sing is the number of iterations for each check
# n_chain is the number of markov chains
def full_gibbs(N_tot, N_sing, n_chain, R, Y, W, X, Z, L, alp_u, beta_u, alp_v, beta_v, sig_y2, a_r, b_r):
    pool = multiprocessing.Pool(processes=n_chain)
    
    # initiate
    args = []
    for i in range(n_chain):
        init = initial_gibbs(Y,W,X,Z,R,L,sig_y2, alp_u, beta_u, alp_v, beta_v, a_r, b_r)
        new_arg = list((N_sing,R,Y,W,X) + init + (alp_u, beta_u, alp_v, beta_v, sig_y2))
        args.append(new_arg)
    results = pool.map(base_gibbs_wrapper, args)

#### initialize a gibbs sampler chain
def initial_gibbs(Y,W,X,Z,R,L, sig_y2, alp_u, beta_u, alp_v, beta_v, a_r, b_r):
    n, m = R.shape
    J = W.shape[1]
    K = X.shape[1]
    
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
    
    return( (Z, U, V, alpha_y, beta_y, sigma_u2, sigma_v2) )
    
def base_gibbs_wrapper(args):
    return base_gibbs(*args)

#### the basic gibbs sampler
# N is the number of iterations
# R is the indicator matrix of observations, dim(R)=(n,m)
# Y is the ratings (binary), dim(Y)=(n,m)
# W is the user feature matrix, dim(W)=(n,J)
# X is the item feature matrix, dim(X)=(m,K)
# Z is the initialized indicators
# U is the initialized noise matrix, dim(U)=(n,L)
# V is the initialized noise matrix, dim(V)=(m,L)
# L is the dimension of U_i and V_j
# alpha_y, beta_y, sigma_u2, sigma_v2 are initial values from prior
# alp_u, beta_u, alp_v, beta_v, sig_y2 are prior parameters (we choose sig_y2 large for a flat prior)
def base_gibbs(N, R, Y, W, X, Z, U, V, alpha_y, beta_y, sigma_u2, sigma_v2, alp_u, beta_u, alp_v, beta_v, sig_y2):
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
    TIMES = []
    LOG_LIKES = []
    
    OBi = []
    for i in range(n):
        observed = [j for j in range(m) if R[i,j]==1.0]
        OBi.append(observed)
    OBj = []
    for j in range(m):
        observed = [i for i in range(n) if R[i,j]==1.0]
        OBj.append(observed)
    
    start_time = time.time()
    
    for nn in range(N):
        if nn%100 == 0:
            now_time = time.time()
            TIMES.append(now_time - start_time)
            start_time = time.time()
        
        #print(nn)
        # sample sigma_u2 and sigma_v2
        sigma_u2 = stats.invgamma.rvs(a= alp_u + n*L/2, scale = beta_u + sum(sum(U*U))/2, size=1)
        sigma_v2 = stats.invgamma.rvs(a= alp_v + m*L/2, scale = beta_v + sum(sum(V*V))/2, size=1)
        SIG_U2.append(sigma_u2[0])
        SIG_V2.append(sigma_v2[0])
        
        # sample alpha_y and beta_y
        sig_alp = np.eye(K)/sig_y2
        sum_r = sum(R)
        for j in range(m):
            sig_alp = sig_alp + sum_r[j] * np.matmul(np.matrix(X[j,]).transpose(), np.matrix(X[j,]))
        L_alp = np.linalg.cholesky(sig_alp)
        Linv_alp = scipy.linalg.solve_triangular(L_alp, np.eye(K), lower=True)
        sig_alp = np.matmul(Linv_alp.transpose(), Linv_alp) #choleskey decomp
        
        sig_beta = np.eye(J)/sig_y2
        sum_rt = sum(R.transpose())
        for i in range(n):
            sig_beta = sig_beta + sum_rt[i] * np.matmul(np.matrix(W[i,]).transpose(), np.matrix(W[i,]))
        #sig_beta = np.linalg.inv(sig_beta)
        L_beta = np.linalg.cholesky(sig_beta)
        Linv_beta = scipy.linalg.solve_triangular(L_beta, np.eye(J), lower=True)
        sig_beta = np.matmul(Linv_beta.transpose(), Linv_beta)
        
        mu_alp = np.matrix(np.zeros(shape=(L,1))).reshape(-1)
        sum_rx = sum(np.array(R) * np.array((Z - np.matmul(W, beta_y.transpose()) - np.matmul(U,V.transpose()))))#save U'V
        for j in range(m):
            mu_alp = mu_alp + sum_rx[j] * X[j,]
        mu_alp = np.matmul(sig_alp, mu_alp.reshape((L,1)))
        
        mu_beta = np.matrix(np.zeros(shape=(L,1))).reshape(-1)
        sum_rw = sum(np.array(R.transpose()) * np.array(Z.transpose() - np.matmul(X, alpha_y.transpose()) - np.matmul(V,U.transpose())))
        for i in range(n):
            mu_beta = mu_beta + sum_rw[i] * W[i,]
        mu_beta = np.matmul(sig_beta, mu_beta.reshape((L,1)))
        
        #print(mu_alp,sig_alp)
        #alpha_y = np.random.multivariate_normal(mean = np.asarray(mu_alp).reshape(-1), cov=sig_alp, size=1)
        stand_alp = np.random.normal(size=K)
        alpha_y = mu_alp.reshape(-1) + np.matmul(Linv_alp.transpose(),stand_alp.reshape((K,1))).reshape(-1)
        #beta_y = np.random.multivariate_normal(mean = np.asarray(mu_beta).reshape(-1), cov=sig_beta, size=1)
        stand_beta = np.random.normal(size=J)
        beta_y = mu_beta.reshape(-1) + np.matmul(Linv_beta.transpose(),stand_beta.reshape((J,1))).reshape(-1)
        ALP_Y.append(alpha_y)
        BETA_Y.append(beta_y)
        
        # sample U_i and V_j
        for i in range(n):
            sig_ui = np.eye(L) / sigma_u2
            mu_ui = np.matrix(np.zeros(shape=(L,1))).reshape(-1)
            for j in OBi[i]:
                sig_ui = sig_ui + R[i,j] * np.matmul(np.matrix(V[j,]).transpose(), np.matrix(V[j,]))
                mu_ui = mu_ui + R[i,j] * (Z[i,j] - sum(sum(W[i,]*np.array(beta_y))) - sum(sum(X[j,]*np.array(alpha_y)))) * V[j,]
            #sig_ui = np.linalg.inv(sig_ui)
            L_ui = np.linalg.cholesky(sig_ui)
            Linv_ui = scipy.linalg.solve_triangular(L_ui, np.eye(L), lower=True)
            sig_ui = np.matmul(Linv_ui.transpose(), Linv_ui)
            mu_ui = np.matmul(sig_ui, mu_ui.reshape(L,1))
            #U[i,] = np.random.multivariate_normal(mean = np.asarray(mu_ui).reshape(-1), cov=sig_ui, size=1)
            stand_ui = np.random.normal(size=L)
            U[i,] = mu_ui.reshape(-1) + np.matmul(Linv_ui.transpose(),stand_ui.reshape((L,1))).reshape(-1)
        
        for j in range(m):
            sig_vj = np.eye(L) / sigma_v2
            mu_vj = np.matrix(np.zeros(shape=(L,1))).reshape(-1)
            for i in OBj[j]:
                sig_vj = sig_vj + R[i,j] * np.matmul(np.matrix(U[i,]).transpose(), np.matrix(U[i,]))
                mu_vj = mu_vj + R[i,j] * (Z[i,j] - sum(sum(W[i,]*np.array(beta_y))) - sum(sum(X[j,]*np.array(alpha_y)))) * U[i,]
            #sig_vj = np.linalg.inv(sig_vj)
            L_vj = np.linalg.cholesky(sig_vj)
            Linv_vj = scipy.linalg.solve_triangular(L_vj, np.eye(L), lower=True)
            sig_vj = np.matmul(Linv_vj.transpose(), Linv_vj)
            mu_vj = np.matmul(sig_vj, mu_vj.reshape(L,1))
            #V[j,] = np.random.multivariate_normal(mean = np.asarray(mu_vj).reshape(-1), cov=sig_vj, size=1)
            stand_vj = np.random.normal(size=K)
            V[j,] = mu_vj.reshape(-1) + np.matmul(Linv_vj.transpose(),stand_vj.reshape((L,1))).reshape(-1)
        
        UU.append(U)
        VV.append(V)
            
        # sample Z_ij
        zmean = np.matmul(W, beta_y.reshape((J,1))) + np.matmul(X, alpha_y.reshape((K,1))).transpose() + np.matmul(U, V.transpose())
        for i in range(n):
            for j in OBi[i]:
                if Y[i,j] == 1:
                    if stats.norm.cdf(zmean[i,j]) > 0.01:
                        Z[i,j] = stats.truncnorm.rvs(-zmean[i,j], np.inf, loc=zmean[i,j],scale=1,size=1)
                else:
                    if stats.norm.cdf(zmean[i,j]) <0.99:
                        Z[i,j] = stats.truncnorm.rvs(-np.inf, -zmean[i,j], loc=zmean[i,j],scale=1,size=1)
        ZZ.append(Z)
        
        # compute likelihood
        log_like = log_likelihood(Y,R,X,W, Z,U,V,sigma_u2,sigma_v2,alpha_y, beta_y, sig_y2, alp_u, alp_v, beta_u, beta_v)
        LOG_LIKES.append(log_like)
        
        
        print(nn)
        print(alpha_y)
        print(beta_y)

                
    
    return SIG_U2, SIG_V2, ALP_Y, BETA_Y, UU, VV, ZZ, LOG_LIKES, TIMES

    
def spec(x, order=2):
    beta, sigma = yule_walker(x,order)
    return sigma**2 / (1. - np.sum(beta))**2

def log_likelihood(Y,R,X,W, Z,U,V,sigma_u2,sigma_v2,alpha_y, beta_y, sig_y2, alp_u, alp_v, beta_u, beta_v):
    n, m = R.shape
    J = W.shape[1]
    K = X.shape[1]
    L = U.shape[1]
    
    ll = 0
    for i in range(n):
        for j in range(m):
            if R[i,j] == 1:
                ll -= 0.5 * (Z[i,j] - sum(sum(W[i,]*np.array(beta_y))) - sum(sum(X[j,]*np.array(alpha_y))) - sum(U[i,]*V[j,]))**2
                ll -= 0.5 * sum(sum(np.array(beta_y)**2))/sig_y2 + 0.5 * sum(sum(np.array(alpha_y)**2))/sig_y2
    ll -= np.log(sigma_u2) * 0.5 * (n*L+2*alp_u+2) + np.log(sigma_v2) * (m*L+2*alp_v+2)*0.5
    ll -= 0.5 * (sum(sum(U**2)) + beta_u) / sigma_u2 + 0.5 * ( sum(sum(V**2)) + beta_v ) / sigma_v2
    
    return(ll[0])

# df is a dataframe of all sampled parameters
def EffectiveSize(df):
    nn, mm = df.shape
    df.columns = ["0"]*mm
    v0 = []
    ESS = []
    for jj in range(mm):
        xx = df.iloc[:,jj]
        xx_mod = AR(xx)
        xx_res = xx_mod.fit(maxlag=100,ic='aic')
        v0.append(xx_res.sigma2/(1.0-sum(xx_res.params))**2)
    for jj in range(mm):
        xx = df.iloc[:,jj]
        ess = xx.std()**2 / v0[jj] * nn
        ESS.append(ess)
    return(ESS)
    
    
    
    
    
    
    
    
    
    