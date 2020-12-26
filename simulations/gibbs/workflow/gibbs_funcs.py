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
def gibbs(N, R, Y, W, X, Z, U, V, alpha_y, beta_y, sigma_u2, sigma_v2, alp_u, beta_u, alp_v, beta_v, sig_y2):
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
    PASS = []
    TIMES = []
    
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
        
        print(nn)
        print(alpha_y)
        print(beta_y)
        # diagnosis every 500 steps
        if nn >0 and nn % 500 == 0:
            # geweke diag 
            first_a = np.zeros((int(0.1*nn),L))
            first_b = np.zeros((int(0.1*nn),L))
            last_a = np.zeros((int(0.5*nn),L))
            last_b = np.zeros((int(0.5*nn),L))
            for ii in range(int(0.1*nn)):
                first_a[ii,] = ALP_Y[ii]
                first_b[ii,] = BETA_Y[ii]
            for ii in range(int(0.5*nn)):
                last_a[ii] = ALP_Y[nn-1-ii]
                last_b[ii] = BETA_Y[nn-1-ii]
            # test for alpha_y
            stats_a = [None]*L
            pass_flag = True
            for jj in range(L):
                f_jj = first_a[:,jj]
                l_jj = last_a[:,jj]
                stat_jj = (f_jj.mean() - l_jj.mean()) / np.sqrt( spec(f_jj)/int(0.1*nn) + spec(f_jj)/int(0.5*nn))
                stats_a[jj] = stat_jj
                if abs(stat_jj) > 1.98:
                    pass_flag = False
            # test for beta_y
            stats_b = [None]*L
            for jj in range(L):
                f_jj = first_b[:,jj]
                l_jj = last_b[:,jj]
                stat_jj = (f_jj.mean() - l_jj.mean()) / np.sqrt( spec(f_jj)/int(0.1*nn) + spec(f_jj)/int(0.5*nn))
                stats_b[jj] = stat_jj
                if abs(stat_jj) > 1.98:
                    pass_flag = False
            PASS.append(pass_flag)
            if pass_flag:
                return SIG_U2, SIG_V2, ALP_Y, BETA_Y, UU, VV, ZZ, TIMES, nn
                
                
    
    return SIG_U2, SIG_V2, ALP_Y, BETA_Y, UU, VV, ZZ, TIMES, nn

    
def spec(x, order=2):
    beta, sigma = yule_walker(x,order)
    return sigma**2 / (1. - np.sum(beta))**2
    
    



