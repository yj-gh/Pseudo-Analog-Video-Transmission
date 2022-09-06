#-*- coding: utf-8 -*-
from re import I
import numpy as np
import matplotlib.pyplot as plt 
import os
import math
import cmath
from PIL import Image
from numpy.core.fromnumeric import mean, reshape
from numpy.lib import average
from scipy.fftpack import dct,idct
from s_modeule.dct import dct2
import random
from numba import jit, njit

# AWGNの作成
# @jit
def make_AWGN(l, r, noise_db, rs_db, av_TX):
    SNR = rs_db - noise_db
    IQ_R_noise = []
    IQ_G_noise = []
    IQ_B_noise = []

    av_noise = av_TX / 10 **(SNR /10)

    for i in range(int((l * r) / 2)):

        Re = np.random.normal(0, np.sqrt(av_noise))
        Im = np.random.normal(0, np.sqrt(av_noise))

        IQ_R_noise.append(Re + Im * 1j)

    for i in range(int((l * r) / 2)):
        Re = np.random.normal(0, np.sqrt(av_noise))
        Im = np.random.normal(0,np.sqrt(av_noise))

        IQ_G_noise.append(Re + Im * 1j)

    for i in range(int((l * r) / 2)):

        Re = np.random.normal(0, np.sqrt(av_noise))
        Im = np.random.normal(0, np.sqrt(av_noise))
       
        IQ_B_noise.append(Re + Im * 1j)
    return IQ_R_noise, IQ_G_noise, IQ_B_noise, SNR

# ノイズの入ったIQ信号に(パスロスとシャードウイングをSNRより)
# @jit
def add_noise(l, r, IQ_R, IQ_G, IQ_B, IQ_R_noise, IQ_G_noise, IQ_B_noise):
    IQ_R_n = []
    IQ_G_n = []
    IQ_B_n = []

    for i in range(int((l * r) / 2)):
        IQ_R_n.append((IQ_R[i].real + IQ_R_noise[i].real) + (IQ_R[i].imag + IQ_R_noise[i].imag) * 1j )
        # # ノイズなし
        # IQ_R_n.append((IQ_R[i].real) + (IQ_R[i].imag) * 1j)

    for i in range(int((l * r) / 2)):
        IQ_G_n.append((IQ_G[i].real + IQ_G_noise[i].real) + (IQ_G[i].imag + IQ_G_noise[i].imag) * 1j )
        # # ノイズなし
        # IQ_G_n.append((IQ_G[i].real) + (IQ_G[i].imag) * 1j)

    for i in range(int((l * r) / 2)):
        IQ_B_n.append((IQ_B[i].real + IQ_B_noise[i].real) + (IQ_B[i].imag + IQ_B_noise[i].imag) * 1j )
        # # ノイズなし
        # IQ_B_n.append((IQ_B[i].real) + (IQ_B[i].imag) * 1j)

    return IQ_R_n, IQ_G_n, IQ_B_n

# 受信電力の変化によるIQ信号の変化
# @njit
def change_chanel(l, r, rs_db, IQ_R_n, IQ_G_n, IQ_B_n):
    IQ_r_R = []
    for i in range(int((l * r) / 2)):
            # # レイリーフェージングの受信電力
            # rf = np.random.rayleigh(1.0)
            # db_rf = 10 * math.log(rf)
            # # I信号，Q信号をそれぞれ変化（ここでのrs_dbはパスロスとシャドーイングの和)
            # IQ_r_R.append((10 ** ((rs_db + db_rf) / 20)) * IQ_R_n[i].real + (10 ** ((rs_db + db_rf) / 20)) * IQ_R_n[i].imag * 1j)
            # レイリーフェージングなし
            IQ_r_R.append((10 ** ((rs_db) / 20)) * IQ_R_n[i].real + (10 ** ((rs_db) / 20)) * IQ_R_n[i].imag * 1j) 

    IQ_r_G = []
    for i in range(int((l * r) / 2)):
            # # レイリーフェージングの受信電力
            # rf = np.random.rayleigh(1.0)
            # db_rf = 10 * math.log(rf)
            # IQ_r_G.append((10 ** ((rs_db + db_rf) / 20)) * IQ_G_n[i].real + (10 ** ((rs_db + db_rf) / 20)) * IQ_G_n[i].imag * 1j)
            # レイリーフェージングなし
            IQ_r_G.append((10 ** ((rs_db) / 20)) * IQ_G_n[i].real + (10 ** ((rs_db) / 20)) * IQ_G_n[i].imag * 1j) 

    IQ_r_B = []
    for i in range(int((l * r) / 2)):
            # # レイリーフェージングの受信電力
            # rf = np.random.rayleigh(1.0)
            # db_rf = 10 * math.log(rf)
            # IQ_r_B.append((10 ** ((rs_db + db_rf) / 20)) * IQ_B_n[i].real + (10 ** ((rs_db + db_rf) / 20)) * IQ_B_n[i].imag * 1j)
            # レイリーフェージングなし
            IQ_r_B.append((10 ** ((rs_db) / 20)) * IQ_B_n[i].real + (10 ** ((rs_db) / 20)) * IQ_B_n[i].imag * 1j) 
    return IQ_r_R, IQ_r_G, IQ_r_B


# 復号に必要なノイズ対角行列の作成に必要な各行のノイズの和の導出(確認)
# @jit
def noise_for_reconstruction(l, r, IQ_R_noise, IQ_G_noise, IQ_B_noise, n):
    noise = np.empty([int(l * r), 3])
    for i in range(int((l * r) / 2)):
        noise[2*i,0] = IQ_R_noise[i].real
        noise[2*i+1,0] = IQ_R_noise[i].imag
    for i in range(int((l * r) / 2)):
        noise[2*i,1] = IQ_G_noise[i].real
        noise[2*i+1,1] = IQ_G_noise[i].imag
    for i in range(int((l * r) / 2)):
        noise[2*i,2] = IQ_B_noise[i].real
        noise[2*i+1,2] = IQ_B_noise[i].imag

    re_noise = noise.reshape((n**2,int((l / n) * (r / n)),3))

    noise_sum = np.empty([n ** 2, 3])  
    for i in range(3):
        noise_sum[:,i] = np.sum(re_noise[:,:,i], axis=1)
    return noise_sum

# IQ信号を元に戻す
# @jit
def IQ_to_matrix(l, r,  IQ_r_R, IQ_r_G, IQ_r_B, n):
    re_HT = np.empty([int(l * r), 3])
    for i in range(int((l * r) / 2)):
        re_HT[2*i,0] = IQ_r_R[i].real
        re_HT[2*i+1,0] = IQ_r_R[i].imag
    for i in range(int((l * r) / 2)):
        re_HT[2*i,1] = IQ_r_G[i].real
        re_HT[2*i+1,1] = IQ_r_G[i].imag
    for i in range(int((l * r) / 2)):
        re_HT[2*i,2] = IQ_r_B[i].real
        re_HT[2*i+1,2] = IQ_r_B[i].imag

    y = re_HT.reshape((n**2,int((l / n) * (r / n)),3))
    return y


# main
def trans(IQ_R, IQ_G, IQ_B, l, r, n, noise_db, rs_db, av_TX):

    IQ_R_noise, IQ_G_noise, IQ_B_noise, SNR = make_AWGN(l, r, noise_db, rs_db, av_TX)

    IQ_R_n, IQ_G_n, IQ_B_n =  add_noise(l, r, IQ_R, IQ_G, IQ_B, IQ_R_noise, IQ_G_noise, IQ_B_noise)

    IQ_r_R, IQ_r_G, IQ_r_B =  change_chanel(l, r, rs_db, IQ_R_n, IQ_G_n, IQ_B_n)

    noise_sum = noise_for_reconstruction(l, r, IQ_R_noise, IQ_G_noise, IQ_B_noise, n)

    y = IQ_to_matrix(l, r,  IQ_r_R, IQ_r_G, IQ_r_B, n)

    return y, int(l / n), int(r / n), noise_sum, SNR