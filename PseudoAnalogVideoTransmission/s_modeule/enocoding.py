#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
import os
import math
from PIL import Image
from numpy.core.fromnumeric import shape
from scipy.fftpack import dct,idct
from s_modeule.dct import dct2
from numba import jit
from s_modeule.scale import scaling


# ピクセルの平均値を取得
# @jit
def px_mean(block):
    mean = []
    for i in range(3):
        mean.append(np.average(block[:,:,i]))
    return mean

# ピクセルの平均値を全ピクセルから引く
# @jit
def sub_px_mean(block, mean):
    block = np.array(block, dtype = "float16")
    for i in range(3):
        block[:,:,i] = block[:,:,i] - mean[i]
    return block

# チャンクの並び替え
# @jit
def sort_chank(n, sq_sum, dct, ch):
    sorted_ch = np.empty([n**2,3])
    sorted_index = np.empty([n**2,3])
    for i in range(3):
        sorted_ch[:,i] = sorted(sq_sum[:,i], reverse=True)
        sorted_index[:,i] = np.argsort(-sq_sum[:,i]) #添字を返す（メタデータ）
    
    #元のチャンクを並び替え
    sorted_index = sorted_index.astype(int)
    ch_sorted = np.empty([n**2,int(dct.shape[0] / n),int(dct.shape[1] /n),3])
    for j in range(3):
        for i in range(len(sorted_index)):
            ch_sorted[i,:,:,j] = ch[sorted_index[i,j],:,:,j]
    return sorted_ch, sorted_index, ch_sorted

# スケーリング係数の乗算
# @jit
def multiplication_scaling(n, dct, ch_sorted, g):
    sc_ch = np.empty([n**2,int(dct.shape[0] / n),int(dct.shape[1] /n),3])
    for j in range(3):
        for i in range(len(ch_sorted)):
            sc_ch[i,:,:,j] = g[i,j] * ch_sorted[i,:,:,j]
    return sc_ch

# アダマール変換
# @jit
def H_T(n, dct, hadamard ,re_sc_ch):
    HT = np.empty((n**2,int((dct.shape[0] /n)*(dct.shape[1] / n)),3))
    for i in range(3):
        HT[:,:,i] = np.dot(hadamard,re_sc_ch[:,:,i])
    return HT

# IQ信号の作成
# @jit
def make_IQ(dct, re_HT):
    IQ_R =[]
    for i in range(int((dct.shape[0] * dct.shape[1]) / 2)):
        IQ_R.append(re_HT[2*i,0] + re_HT[2*i+1,0]*1j)
    IQ_G =[]
    for i in range(int((dct.shape[0] * dct.shape[1]) / 2)):
        IQ_G.append(re_HT[2*i,1] + re_HT[2*i+1,1]*1j)
    IQ_B =[]
    for i in range(int((dct.shape[0] * dct.shape[1]) / 2)):
        IQ_B.append(re_HT[2*i,2] + re_HT[2*i+1,2]*1j)
    return IQ_R, IQ_G, IQ_B


def encode(block, n ,P):
    # ピクセルの平均値を取得
    mean = px_mean(block)
    pixel = sub_px_mean(block, mean)

    # DCT係数に変換
    dct = dct2(pixel)

    ch = np.concatenate(np.hsplit(np.stack(np.hsplit(dct,n)), n))
     # 各チャンクの要素を２乗
    sq = np.power(ch,2)

    # 各チャンクの要素の２乗にした値の和を求める
    sq_sum = np.sum(sq,axis=1)
    sq_sum = np.sum(sq_sum,axis=1) #rgbそれぞれの和，この状態でRGBごとに並び替え
    # チャンクの並び替え
    sorted_ch, sorted_index, ch_sorted = sort_chank(n, sq_sum, dct, ch)

    #スケーリング係数の決定
    g =scaling(sorted_ch, n, P)

    # スケーリング係数の乗算
    sc_ch = multiplication_scaling(n, dct, ch_sorted, g)

    # ３次元配列を２次元配列に
    re_sc_ch = sc_ch.reshape(n**2,int((dct.shape[0] /n)*(dct.shape[1] / n)),3)

    # アダマール行列の生成
    ha =  np.array([[1,1],[1,-1]])

    def sylvester(n):
        b = np.array(([1,1],[1,-1]))
        for i in range(n-1):
            b = np.kron(ha,b)
        return b
    hadamard = sylvester(int(math.log2(n**2)))

    # アダマール変換
    HT = H_T(n, dct, hadamard ,re_sc_ch)
    re_HT = HT.reshape(int(dct.shape[0] * dct.shape[1]),3)

    IQ_R, IQ_G, IQ_B = make_IQ(dct, re_HT)

    return IQ_R, IQ_G, IQ_B, dct.shape[0], dct.shape[1], sorted_ch, sorted_index, mean, HT





# def encode(block,n, P):  
    # ピクセルの平均値
    mean = []
    for i in range(3):
        mean.append(np.average(block[:,:,i]))

    block = np.array(block, dtype = "float16")
    for i in range(3):
        block[:,:,i] = block[:,:,i] - mean[i]

    # ２次元離散コサイン変換
    c = dct2(block) 

     # 各チャンクの要素を２乗
    sq = np.power(ch,2)

    # 各チャンクの要素の２乗にした値の和を求める
    sq_sum = np.sum(sq,axis=1)
    sq_sum = np.sum(sq_sum,axis=1) #rgbそれぞれの和，この状態でRGBごとに並び替え

    
    # チャンクに分割 c1=np.concatenate(np.hsplit(np.stack(np.hsplit(c,横分割数)), 縦分割数))
    ch = np.concatenate(np.hsplit(np.stack(np.hsplit(c,n)), n))

    # 各チャンクの要素を２乗
    sq = np.power(ch,2)

    # 各チャンクの要素の２乗にした値の和を求める
    sq_sum = np.sum(sq,axis=1)
    sq_sum = np.sum(sq_sum,axis=1) #rgbそれぞれの和，この状態でRGBごとに並び替え

    # チャンクを並び替え
    sorted_ch = np.empty([n**2,3])
    sorted_index = np.empty([n**2,3])
    for i in range(3):
        sorted_ch[:,i] = sorted(sq_sum[:,i], reverse=True)
        sorted_index[:,i] = np.argsort(-sq_sum[:,i]) #添字を返す（メタデータ）
    
    # 元のチャンクを並び替え
    sorted_index = sorted_index.astype(int)
    ch_sorted = np.empty([n**2,int(c.shape[0] / n),int(c.shape[1] /n),3])
    for j in range(3):
        for i in range(len(sorted_index)):
            ch_sorted[i,:,:,j] = ch[sorted_index[i,j],:,:,j] 

    # # 各チャンクの分散の導出（メタデータ）
    # va = np.empty([n**2,3])
    # for j in range(3):
    #     for i in range(len(ch_sorted)):
    #         va[i,j] = np.var(abs(ch_sorted[i,:,:,j])) 
    
    # # 高周波チャンクの削除
    # for i in range(63):
    #     ch_sorted[ch_sorted.shape[0]-1-i,:,:,:] = 0.0
    #     # sorted_ch[sorted_ch.shape[0]-1-i,:] = 0.0
    # # 低周波チャンクの削除
    # for i in range(1):
    #     ch_sorted[i,:,:,:] = 0.0

    # スケーリング係数の決定
    sq_sorted_ch = sorted_ch ** 0.5
    sd = []
    for i in range(3): 
        sd.append(np.sum(sq_sorted_ch[:,i]))

    g = np.empty([n**2,3])
    for j in range(3):
        # sd = 0.0
        # for i in range(1):
        for i in range(len(sorted_ch)):
            # sd += sorted_ch[i,j] ** 0.5
            g[i,j] = sorted_ch[i,j] ** (-1/4) * (P / sd[j]) ** 0.5

    # np.savetxt('./g.txt', g)
    
    # スケーリング係数を乗算
    sc_ch = np.empty([n**2,int(c.shape[0] / n),int(c.shape[1] /n),3])
    for j in range(3):
        for i in range(len(ch_sorted)):
            sc_ch[i,:,:,j] = g[i,j] * ch_sorted[i,:,:,j]
    
    sq_sc_ch = np.power(sc_ch,2)
    qed = np.sum(sq_sc_ch,axis=0)
    qed = np.sum(qed,axis=0)
    qed = np.sum(qed,axis=0)##送信電力の割り当てが成功しているかの確認
    
    # ３次元配列を２次元配列に
    re_sc_ch = sc_ch.reshape(n**2,int((c.shape[0] /n)*(c.shape[1] / n)),3)

    # アダマール行列の生成
    ha =  np.array([[1,1],[1,-1]])

    def sylvester(n):
        b = np.array(([1,1],[1,-1]))
        for i in range(n-1):
            b = np.kron(ha,b)
        return b
    hadamard = sylvester(int(math.log2(n**2)))

    # アダマール行列の乗算
    HT = np.empty((n**2,int((c.shape[0] /n)*(c.shape[1] / n)),3))
    for i in range(3):
        HT[:,:,i] = np.dot(hadamard,re_sc_ch[:,:,i])

    
    # IQ信号に
    re_HT = HT.reshape(int(c.shape[0] * c.shape[1]),3)

    IQ_R =[]
    for i in range(int((c.shape[0] * c.shape[1]) / 2)):
        IQ_R.append(re_HT[2*i,0] + re_HT[2*i+1,0]*1j)
    IQ_G =[]
    for i in range(int((c.shape[0] * c.shape[1]) / 2)):
        IQ_G.append(re_HT[2*i,1] + re_HT[2*i+1,1]*1j)
    IQ_B =[]
    for i in range(int((c.shape[0] * c.shape[1]) / 2)):
        IQ_B.append(re_HT[2*i,2] + re_HT[2*i+1,2]*1j)
    
    return IQ_R, IQ_G, IQ_B, c.shape[0], c.shape[1],  c, sorted_ch, sorted_index, mean, HT