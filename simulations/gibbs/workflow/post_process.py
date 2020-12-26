#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:41:03 2020

@author: ying
"""

import pandas as pd
import numpy as np
import os.path
from os import path

#os.chdir("/Users/ying/Desktop/Stanford/Research/bayes/cluster_compute")


df_res = pd.DataFrame()

ns = [100, 500, 1000, 5000]
ms = [20, 100, 200, 1000]
count=0

for coef_scale in [1.0, 2.0, 5.0]:
    for nid in [0, 1, 2,3]:
        for mid in [0,1,2,3]:
            for L in [2,5,10]:
                pathname = "res_scale_"+str(coef_scale)+"n_"+str(nid)+"m_"+str(mid)+"L_"+str(L)+".csv"
                if path.exists(pathname):
                    df_snr = pd.read_csv("snr_scale_"+str(coef_scale)+"n_"+str(nid)+"m_"+str(mid)+"L_"+str(L)+".csv")
                    snr = df_snr.iloc[0,1]
                    
                    df_alpy = pd.read_csv("alpy_scale_"+str(coef_scale)+"n_"+str(nid)+"m_"+str(mid)+"L_"+str(L)+".csv")
                    err_alpy = (df_alpy.iloc[1:,1:] -df_alpy.iloc[0,1:]).mean()
                    err_alpy = np.sqrt(np.power(err_alpy,2).mean())
                    
                    df_betay = pd.read_csv("betay_scale_"+str(coef_scale)+"n_"+str(nid)+"m_"+str(mid)+"L_"+str(L)+".csv")
                    err_betay = (df_betay.iloc[1:,1:] -df_betay.iloc[0,1:]).mean()
                    err_betay = np.sqrt(np.power(err_betay,2).mean())
                    
                    df_sigu = pd.read_csv("sigu_scale_"+str(coef_scale)+"n_"+str(nid)+"m_"+str(mid)+"L_"+str(L)+".csv")
                    err_sigu = (df_sigu.iloc[:,1]- 1).mean()
                    
                    df_sigv = pd.read_csv("sigv_scale_"+str(coef_scale)+"n_"+str(nid)+"m_"+str(mid)+"L_"+str(L)+".csv")
                    err_sigv = (df_sigv.iloc[:,1]- 1).mean()
                    df_res = pd.concat([df_res, pd.DataFrame(np.matrix([coef_scale, ns[nid], ms[mid], L, err_sigu, err_sigv, err_alpy, err_betay ]).reshape(-1))] )

df_res.columns = ["scale","n","m","L","sigu_err","sigv_err","alp_err","beta_err"]

df_res.to_csv("allres.csv")