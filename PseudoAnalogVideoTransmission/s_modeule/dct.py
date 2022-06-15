#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
import os
from PIL import Image
from scipy.fftpack import dct, idct

#2D DCT
def dct2(block):
   
    R = block[:,:,0]
    G = block[:,:,1]
    B = block[:,:,2]
    dct_image = np.zeros(block.shape)
    R = dct(dct(R, axis=0, norm='ortho'), axis=1, norm='ortho')
    G = dct(dct(G, axis=0, norm='ortho'), axis=1, norm='ortho')
    B = dct(dct(B, axis=0, norm='ortho'), axis=1, norm='ortho')
    # np.savetxt('./R.txt', R)
    # np.savetxt('./G.txt', G)
    # np.savetxt('./B.txt', B)
    dct_image[:,:,0]=R
    dct_image[:,:,1]=G
    dct_image[:,:,2]=B
    
    return dct_image

#2D逆DCT(IDCT)
def idct2(block):
    
    R = block[:,:,0]
    G = block[:,:,1]
    B = block[:,:,2]
    idct_image = np.zeros(block.shape)
    R = idct(idct(R, axis=0, norm='ortho'), axis=1, norm='ortho')
    G = idct(idct(G, axis=0, norm='ortho'), axis=1, norm='ortho')
    B = idct(idct(B, axis=0, norm='ortho'), axis=1, norm='ortho')
    idct_image[:,:,0]=R
    idct_image[:,:,1]=G
    idct_image[:,:,2]=B

    return idct_image