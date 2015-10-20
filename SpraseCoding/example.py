# coding:utf-8
import random
import math
import numpy as np
import scipy.io
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import logging
logging.basicConfig(level=logging.WARNING)
import sparse_coding as sc

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
    plt.imshow(collage, cmap=cm.gray)

def callback(X, B, S):
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

#images = scipy.io.loadmat("IMAGES.mat")["IMAGES"]
images = misc.lena() #获取一张图片
patch_size = 8 #获取patch的尺寸
num_patches = 4 #获取patch的数量
columns = []
for i in xrange(num_patches): #0、1、2、...、num_patches循环
    #j = random.randint(0, images.shape[2] - 1)  #任意的选择一个通道
    y, x = [random.randint(0, images.shape[d] - patch_size) for d in (0, 1)] #任意的产生一组坐标
    #column = images[x:x+patch_size, y:y+patch_size, j].reshape((patch_size**2, 1)) #提取某一通道上的patch
    column = images[x:x+patch_size, y:y+patch_size].reshape((patch_size**2, 1)) #提取patch，单通道不需要j，并且变为列向量
    columns.append(column)
X = np.hstack(columns)
X = np.asarray(X,dtype='float64')
print (X.shape,X.dtype)
# test callback function on svd
#svd = np.linalg.svd(X, full_matrices=False)
#print [x.shape for x in svd]
#callback(X, svd[0], np.dot(np.diag(svd[1]), svd[2]))

num_bases = 64#要产生基的个数
sc.sparse_coding(X, num_bases, 0.4, 100, lambda B, S: callback(X, B, S))
