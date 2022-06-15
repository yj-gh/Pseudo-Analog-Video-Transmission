#-*- coding: utf-8 -*-
from s_modeule.dct import idct2
from s_modeule.hadamard import hadamard
import numpy as np
import matplotlib.pyplot as plt 
import os
import math
from PIL import Image
from scipy.fftpack import dct,idct
from s_modeule.scale import scaling
from numba import jit

# @jit
def make_A(n, variance):
    A = np.empty([n**2,n**2,3])
    for i in range(3):
        A[:,:,i] = np.diag(variance[:,i]) #分散値
    return A

# @jit
def make_G(n, g):
    G = np.empty([n**2,n**2,3])
    for i in range(3):
        G[:,:,i] = np.diag(g[:,i]) #スケーリング係数
    return G

# @jit
def make_N(n, noise):
    N = np.empty([n**2,n**2,3])
    for i in range(3):
        N[:,:,i] = np.diag(noise[:,i]) #スケーリング係数
    return N

# @jit
def make_C(n, ha, G):
    C = np.empty([n**2,n**2,3]) 
    for i in range(3):   
        C[:,:,i] = np.dot(ha, G[:,:,i]) #アダマール行列とスケーリング係数の乗算値
    return C

# @jit
def make_CT(n, C):
    CT =np.empty([n**2,n**2,3])
    for i in range(3):
        CT[:,:,i] = np.transpose(C[:,:,i]) #Cの転置行列
    return CT

# @jit
def make_CA(n, C, A):
    CA = np.empty([n**2,n**2,3])
    for i in range(3):
        CA[:,:,i] = np.dot(C[:,:,i], A[:,:,i])
    return CA

# @jit
def make_CACT(n, CA, CT):
    CACT =np.empty([n**2,n**2,3])
    for i in range(3):
        CACT[:,:,i] = np.dot(CA[:,:,i], CT[:,:,i])
    return CACT

# @jit
def make_CACT_N_inv(n, CACT):
    CACT_N_inv = np.empty([n**2,n**2,3])
    for i in range(3):   
        CACT_N_inv[:,:,i] = np.linalg.inv(CACT[:,:,i]) #CACTの逆行列
    return CACT_N_inv

# @jit
def make_ACT(n, A, CT):
    ACT = np.empty([n**2,n**2,3])
    for i in range(3):
        ACT[:,:,i] = np.dot(A[:,:,i], CT[:,:,i])
    return ACT

# @jit
def make_ACT_CACT_N_inv(n, ACT, CACT_N_inv):
    ACT_CACT_N_inv = np.empty([n**2,n**2,3])
    for i in range(3):
        ACT_CACT_N_inv[:,:,i] = np.dot(ACT[:,:,i], CACT_N_inv[:,:,i])
    return ACT_CACT_N_inv

# @jit
def restore_X(n, l, r, ACT_CACT_N_inv, y_n):
    X = np.empty((n**2, l * r,3))
    for i in range(3):
        X[:,:,i] = np.dot(ACT_CACT_N_inv[:,:,i], y_n[:,:,i])
    return X

# チャンクを元の並びに戻す
# @jit
def reassemble_ch(X, l, r, pos):
    chank_X = np.empty([X.shape[0], l, r,3])
    for j in range(3):
        for i in range(X.shape[0]):
            chank_X[i,:,:,j]=X[i,:,j].reshape( l, r) #sorted chと同じ

    o_ch = np.empty([X.shape[0], l, r,3])
    for j in range(3):
        for i in range(X.shape[0]):
            o_ch[pos[i,j],:,:,j] = chank_X[i,:,:,j]
    return o_ch

# dct係数を画素値に直す
# @jit
def restore_dct(o_ch, n):
    # チャンクを元の行列に戻す
    re_dct = o_ch[0]
    for i in range(n-1):
        re_dct = np.concatenate((re_dct,o_ch[i+1]),axis = 1)

    for j in range(n-1):
        re_dct1 = o_ch[n*(j+1)]
        for i in range(n-1):
            re_dct1 = np.concatenate((re_dct1,o_ch[n*(j+1)+i+1]),axis = 1)  
        re_dct = np.concatenate([re_dct,re_dct1])
    return re_dct

# ピクセルの平均値を足す
# @jit
def add_px_mean(re_c, px_mean):
    for i in range(3):
        re_c[:,:,i] = re_c[:,:,i] + px_mean[i]
    return re_c

#0-255のRGBに戻す
# @jit
def correct_px(re_c):
    for k in range(re_c.shape[2]):
        for j in range(re_c.shape[1]):
            for i in range(re_c.shape[0]):
                if re_c[i,j,k] > 255:
                    re_c[i,j,k] = 255
                elif re_c[i,j,k] < 0:
                    re_c[i,j,k] = 0
                else:
                    re_c[i,j,k] = re_c[i,j,k]

    re_img = re_c.astype(np.uint8)
    return re_img


def decode(variance, pos, n, y_n, l, r, P, noise, rs_db, px_mean):
    # 電波マップによる修正
    err = 0
    # 電波マップの誤差
    rs_db += err
    # channel係数で信号の修正
    y_n = (10 ** ((-rs_db)/ 20 )) * y_n

    A =  make_A(n, variance)

    g = scaling(variance, n, P)
    G =  make_G(n, g)

    N = make_N(n, noise)

    ha = hadamard(int(math.log2(n**2)))

    C = make_C(n, ha, G)
    CT = make_CT(n, C)
    CA = make_CA(n, C, A)
    CACT = make_CACT(n, CA, CT)
    CACT_N = CACT + N
    CACT_N_inv = make_CACT_N_inv(n, CACT)
    ACT = make_ACT(n, A, CT)
    ACT_CACT_N_inv = make_ACT_CACT_N_inv(n, ACT, CACT_N_inv)
    X = restore_X(n, l, r, ACT_CACT_N_inv, y_n)

    o_ch = reassemble_ch(X, l, r, pos)

    re_dct = restore_dct(o_ch, n)
    re_c = idct2(re_dct)
    re_c = add_px_mean(re_c, px_mean)

    re_img = correct_px(re_c)

    return re_img