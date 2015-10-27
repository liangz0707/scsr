# coding:utf-8
__author__ = 'liangz14'

import math
import numpy as np
from skimage import color

from scipy.misc  import imresize

from skimage import data

import matplotlib.pyplot as plt
#bicubic params
def S(x,a=-0.5):
    if x<1 and x>=0:
        return (a+2)*np.abs(x)**3-(a+3)*np.abs(x)**2+1
    if x<2 and x>=1:
        return a*np.abs(x)**3-5*a*np.abs(x)**2+8*a*np.abs(x)-4*a
    else:
        return 0

'''
    this method is for 2d array in float 三线性差值是使用最近的16个店进行插值
    src_img :float image 浮点数图像
    scale: float scale for x and y 浮点数倍数
'''

def bicubic2d(src_img,scale):
    if src_img.shape[0]<4:
        return imresize(src_img,scale,'bicubic')
    scale = scale*1.0  #we need a float scale
    src_img = np.asarray(src_img,dtype ='float64') #we need a float 2d array
    xlen = src_img.shape[1]
    ylen = src_img.shape[0]
    xlenh = int(xlen * scale)
    ylenh = int(ylen * scale)
    dst_img = np.zeros((ylenh, xlenh))
    for i in range(ylenh):
        for j in range(xlenh):
            #we find the nearest 16 points for every points in dst image
            tmp_i = i/scale
            tmp_j = j/scale
            src_i = np.floor(tmp_i)
            src_j = np.floor(tmp_j)

            v = tmp_i - src_i
            u = tmp_j - src_j
            #范围限定
            if src_i == 0:
                src_i = 1
            if src_j == 0:
                src_j = 1
            if src_j+3 - xlen >0 :
                src_j = xlen - 3
            if src_i+3 - ylen >0 :
                src_i = ylen - 3

            patch = src_img[src_i-1:src_i+3 ,src_j-1 :src_j+3]

            A = np.asarray([S(1+v),S(v),S(1-v),S(2-v)],dtype='float64')
            B = np.asarray([S(1+u),S(u),S(1-u),S(2-u)],dtype='float64')

            tempA = np.dot(A,patch)
            dst_img[i,j] = np.dot(tempA,B)
    return  dst_img

def psnr( img1,img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 20 * math.log10(np.max(img1) / math.sqrt(mse))

def main():
    src_rgb_img = data.lena()
    #src_rgb_img = io.imread('Child_input.png')

    src_Lab_img = color.rgb2lab(src_rgb_img)
    src_Lab_img_L = src_Lab_img[:,:,0]

    src_Lab_img_a = src_Lab_img[:,:,1]
    src_Lab_img_b = src_Lab_img[:,:,2]

    #down_sampling with scale = 0.5
    scale = 0.25
    src = bicubic2d(src_Lab_img_L,scale)*2.55

    #up_sampling with scale = 2.0 to source image by different method
    dst_me = bicubic2d(src,4.0)
    dst_other = imresize(src,4.0,interp='bicubic')#this method result is int and in [0,255]

    #evalue different with PSNR MY result is better
    print psnr(dst_me/2.55,src_Lab_img_L)
    print psnr(dst_other/2.55,src_Lab_img_L)

    plt.imshow(dst_me,interpolation="none")
    plt.show()

