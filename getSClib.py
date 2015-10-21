# coding:utf-8
__author__ = 'liangz14'

# coding:utf-8
import random
import math
import numpy as np
import scipy.io
import skimage.io
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import SpraseCoding.sparse_coding as sc
import os
from scipy.misc  import imresize
from  scipy.signal import convolve2d

def onto_unit(x):
    a = np.min(x)
    b = np.max(x)
    return (x - a) / (b - a)

def visualize_patches(B):
    # assume square
    mpatch = int(math.floor(math.sqrt(B.shape[0])))
    npatch = mpatch

    m = int(math.floor(math.sqrt(B.shape[1])))
    n = int(math.ceil(B.shape[1] * 1.0 / m))
    collage = np.zeros((m*mpatch, n*npatch))
    for i in xrange(m):
        for j in xrange(n):
            try:
                patch = B[:, i*n + j]
            except IndexError:
                continue
            patch = onto_unit(patch.reshape((mpatch, npatch)))
            collage[i*mpatch:(i+1)*mpatch, j*npatch:(j+1)*npatch] = patch


def callback(X, B, S):
    pass
    '''
    plt.subplot(2, 2, 1)
    visualize_patches(X)
    plt.title("originals")
    plt.subplot(2, 2, 2)
    visualize_patches(B)
    plt.title("bases")
    plt.subplot(2, 2, 3)
    visualize_patches(np.dot(B, S))
    plt.title("reconstructions")
    plt.subplot(2, 2, 4)
    visualize_patches(X - np.dot(B, S))
    plt.title("differences")
    plt.show()
    '''


def getTrainSet(img_lib,patch_dic_num):
    img_num = len(img_lib)-1 #所需的patch字典个数 HR-LR
    #抽取字典~这个字典应该是HR-LRduo堆叠起来的结果
    scale = 4.0 #放大倍数
    feat_scale=2.0
    patch_sizel = 3 #高清patch的尺寸， 对应底分辨率的patch就是   patch_size/scale
    patch_sizeh = patch_sizel*scale
    patch_sizem = patch_sizel*feat_scale

    HRpatch_lib = []
    F_1_lab = []
    F_2_lab = []
    F_3_lab = []
    F_4_lab = []
    H_lib = []

    f1=np.asarray([[-1,0,1]],dtype='float64')
    f2=np.asarray([[-1],[0],[1]],dtype='float64')
    f3=np.asarray([[1,0,-2,0,1]],dtype='float64')
    f4=np.asarray([[1],[0],[-2],[0],[1]],dtype='float64')

    for i in range(patch_dic_num):
        img_i =random.randint(0,img_num) #得到去那一张图片。
        img_temp = img_lib[img_i]
        y,x = [random.randint(0,img_temp.shape[d] - patch_sizeh) for d in (0,1)]
        HRpatch = img_temp[y:y+patch_sizeh,x:x+patch_sizeh]
        #高清patch
        HRpatch_lib.append(HRpatch)
        LRpatch = imresize(HRpatch,0.25,'bicubic')
        LRpatch = imresize(LRpatch,2.0,'bicubic')
        #低分辨率特征patch
        F_1_lab.append(convolve2d(LRpatch,f1,mode='same'))
        F_2_lab.append(convolve2d(LRpatch,f2,mode='same'))
        F_3_lab.append(convolve2d(LRpatch,f3,mode='same'))
        F_4_lab.append(convolve2d(LRpatch,f4,mode='same'))
    ylist = []
    xlist = []
    for i in range(len(HRpatch_lib)):
        H=HRpatch_lib[i]
        F=np.zeros((patch_sizem,patch_sizem,4))
        F[:,:,0]= convolve2d(F_1_lab[i],f1,mode='same')
        F[:,:,1]= convolve2d(F_2_lab[i],f2,mode='same')
        F[:,:,2]= convolve2d(F_3_lab[i],f3,mode='same')
        F[:,:,3]= convolve2d(F_4_lab[i],f4,mode='same')
        yy=[]
        normalization_m = math.sqrt(np.sum(F**2))
        if normalization_m > 1:
            yy = F/normalization_m
        else:
            yy = F

        xx=H/np.mean(H)

        ylist.append(yy.reshape((patch_sizeh*patch_sizeh)))
        xlist.append(xx.reshape((patch_sizem*patch_sizem*4)))
    return (xlist,ylist)

#读取目录下的图片:提取全部图片，并提取成单通道图像，并归一化数值[0,1]
def readImgTrain(cur_dir):
    img_file_list =os.listdir(cur_dir) #读取目录下全部图片文件名
    img_lib = []
    for file_name in img_file_list:
        full_file_name = os.path.join(cur_dir,file_name)
        img_tmp = skimage.io.imread(full_file_name) #读取一张图片
        if img_tmp.ndim !=2:
            img_tmp =  skimage.color.rgb2lab(img_tmp)
            img_tmp = img_tmp[:,:,0]
            img_tmp = img_tmp /100
        else:
            img_tmp = skimage.img_as_float(img_tmp)
        #print img_tmp.dtype , img_tmp.shape , np.max(img_tmp),np.min(img_tmp)
        img_lib.append(img_tmp)
    return img_lib
#得到img_lib
img_lib = readImgTrain('D:\workspace\scsr\TrainData')
#得到训练的开始和结果
(xlist,ylist) = getTrainSet(img_lib,10000)

columns = []
for i in range(len(xlist)):
    p = (xlist[i]+ylist[i]).reshape(len(xlist[i]+ylist[i]))
    p = np.zeros((len(xlist[i])+len(ylist[i]),1))
    print p.shape
    p[:len(xlist[i]),0] = xlist[i]
    p[len(xlist[i]):,0] = ylist[i]
    print xlist[i].shape,ylist[i].shape,p.shape
    columns.append(p)
X = np.hstack(columns)
X = np.asarray(X,dtype='float64')
print (X.shape,X.dtype)

num_bases = 512#要产生基的个数
(B, S) = sc.sparse_coding(X, num_bases, 0.4, 100, lambda B, S: callback(X, B, S))
from sklearn.externals import joblib
joblib.dump(B, 'Base_lib.pkl')
joblib.dump(S, 'S_lib.pkl')