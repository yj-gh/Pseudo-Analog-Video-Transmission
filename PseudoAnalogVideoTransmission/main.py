#-*- coding: utf-8 -*-
from numpy.core.fromnumeric import mean
from numpy.lib import average
from s_modeule.decoding import decode
from s_modeule.channel import trans
from s_modeule.enocoding import encode
import numpy as np
import matplotlib.pyplot as plt 
import os
from PIL import Image
from scipy.fftpack import dct,idct
import math
import shutil
from numba import jit

@jit
def get_PSNR(row, line, img, re_img):
  SE = 0
  for k in range(3):
      for j in range(row):
          for i in range(line):
              SE += (int(img[i,j,k]) - int(re_img[i,j,k])) ** 2
  
  MSE = SE / (3*line * row)
  RMSE = MSE ** 0.5
  PSNR = 20 * math.log10(255 / (MSE ** 0.5))
  return PSNR


if __name__=="__main__":
  #フォルダ名
  dir_name = "images"
  #指定フォルダ内のファイル名をすべて取得
  file_path = os.listdir(path=dir_name)
  file_path = sorted(file_path)

  for path in file_path:
        #指定したパスのファイルを配列で取得（ここでは画像）
        img = np.array(Image.open(os.path.join(dir_name,path)))
        name = os.path.splitext(path)
        
        n = 8 #分割数

        # 送信電力総和
        P = 1 * img.shape[0] * img.shape[1]
    
        # ノイズフロアの測定値
        noise_db = -120
        
        # 受信電力
        r_s_db = -95

        # 変調(v, posi, px_meanはメタデータ)
        R, G, B, line, row, v, posi, px_mean, y = encode(img, n, P)

        # 無線部
        y_n, line_ch, row_ch, noise, SNR = trans(R, G, B, line, row, n, noise_db, r_s_db)

        
        # 復調
        re_img = decode(v, posi, n, y_n, line_ch, row_ch, P, noise, r_s_db, px_mean)


        # PSNRを求める
        PSNR = get_PSNR(row, line, img, re_img)

        SNR = round(SNR)
        PSNR = round(PSNR)

        plt.subplot(1,2,1)
        plt.imshow(img,cmap="Greys")
        plt.title("original")
        plt.xticks([])
        plt.yticks([])
        
        plt.subplot(1,2,2)
        plt.imshow(re_img,cmap="Greys")
        plt.title("restored")
        plt.xlabel("SNR="+ str(SNR) + ", PSNR=" + str(PSNR))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        # break
