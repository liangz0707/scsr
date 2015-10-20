# coding:utf-8
__author__ = 'liangz14'

import math
import numpy as np
from skimage import io
from skimage import color
import scipy.io
from scipy.misc  import imresize
from scipy.signal import convolve2d
from sklearn.externals import joblib
#求解线性方程组：
from sklearn import linear_model

import matplotlib.pyplot as plt
#超分辨率处理函数
def SR(img,Dh, Dl,l_da=0.0001, scale=3.0,patch_sizel=3.0, overlap=1.0 ,img_size=1.0):
    #l_da是经验值

    #得到特征域
    feat_scale=2.0
    patch_sizeh = patch_sizel*scale
    patch_sizem = patch_sizel*feat_scale

    imgm = imresize(img, feat_scale,'bicubic')
    imgh = np.zeros((img.shape[0]*3,img.shape[1]*3))
    normalizationMat = np.zeros((img.shape[0]*3,img.shape[1]*3))

    print imgm.shape , imgh.shape
    f1=np.asarray([[-1,0,1]],dtype='float64')
    f2=np.asarray([[-1],[0],[1]],dtype='float64')
    f3=np.asarray([[1,0,-2,0,1]],dtype='float64')
    f4=np.asarray([[1],[0],[-2],[0],[1]],dtype='float64')
    F=np.zeros((imgm.shape[0],imgm.shape[1],4))


    F[:,:,0]= convolve2d(imgm,f1,mode='same')
    F[:,:,1]= convolve2d(imgm,f2,mode='same')
    F[:,:,2]= convolve2d(imgm,f3,mode='same')
    F[:,:,3]= convolve2d(imgm,f4,mode='same')

    xgrid = np.ogrid[int(patch_sizel/2.0):img.shape[1]-patch_sizel:patch_sizel - overlap]
    ygrid = np.ogrid[int(patch_sizel/2.0):img.shape[0]-patch_sizel:patch_sizel - overlap]
    xgridm = (xgrid-1)*feat_scale +1
    ygridm = (ygrid-1)*feat_scale+1
    xgridh = (xgrid-1)*scale+1
    ygridh = (ygrid-1)*scale+1


    for i in range(len(xgridm)):
        for j in  range(len(ygridm)):
            x,y = xgridm[i],ygridm[j]
            xh,yh = xgridh[i],ygridh[j]

            patchm = imgm[y:y+patch_sizem, x:x+patch_sizem]

            avg_patch = np.mean(patchm)
            featpatch = np.transpose(F[y:y+patch_sizem, x:x+patch_sizem,:],axes=(2,1,0)).reshape(patch_sizem*patch_sizem*4)
            normalization_m = math.sqrt(np.sum(featpatch**2))

            yy=[]
            if normalization_m > 1:
                yy = featpatch/normalization_m
            else:
                yy = featpatch
            clf = linear_model.Lasso(alpha=l_da)
            clf.fit(Dl, yy)
            w = clf.coef_
            #print x,y,np.sum(np.abs(w))
            patchh=[]
            if normalization_m > 1:
                    patchh = np.dot(Dh,w)*normalization_m
            else:
                    patchh = np.dot(Dh,w)

            patchh = patchh.reshape([patch_sizeh, patch_sizeh]) + avg_patch
            patchh = np.transpose(patchh)

            imgh[yh:yh+patch_sizeh, xh:xh+patch_sizeh]= imgh[yh:yh+patch_sizeh, xh:xh+patch_sizeh] + patchh
            normalizationMat[yh:yh+patch_sizeh, xh:xh+patch_sizeh]= normalizationMat[yh:yh+patch_sizeh, xh:xh+patch_sizeh]+ 1
            print x,y
    imgh = imgh/normalizationMat
    joblib.dump(imgh, 'img_h2.pkl')
    return imgh
def readSR():
    return joblib.load('img_h2```````.pkl')

#加载字典
mat = scipy.io.loadmat('./Dictionary/dictionary.mat')
Dl_dict = np.asarray( mat['Dl'],dtype='float64')
Dh_dict = np.asarray( mat['Dh'],dtype='float64')
#print sc_dict.shape

#提取出L通道进行计算其他通道使用Bicubic进行计算
src_rgb_img = io.imread('./Data/Child_input.png')
#src_rgb_img = src_rgb_img[30:70,22:60,:]
print src_rgb_img.shape
#需要变称散的倍数才能处理
src_rgb_img = src_rgb_img[:src_rgb_img.shape[0]-src_rgb_img.shape[0]%3,:src_rgb_img.shape[1]-src_rgb_img.shape[0]%3,:]
#print src_rgb_img.shape

src_Lab_img = color.rgb2lab(src_rgb_img)
src_Lab_img_L = src_Lab_img[:,:,0]
src_Lab_img_a = src_Lab_img[:,:,1]
src_Lab_img_b = src_Lab_img[:,:,2]

dst_Lab_img_L = readSR()
#dst_Lab_img_L = SR(src_Lab_img_L,Dh_dict,Dl_dict)
#dst_Lab_img_L = imresize(src_Lab_img_L,3.0,'bicubic')
print np.max(src_Lab_img_L),np.min(src_Lab_img_L)
dst_Lab_img_L= dst_Lab_img_L/255.0*(np.max(src_Lab_img_L)-np.min(src_Lab_img_L))+np.min(src_Lab_img_L)

dst_Lab_img_a = imresize(src_Lab_img_a,3.0,'bicubic')
print np.max(src_Lab_img_a),np.min(src_Lab_img_a)
dst_Lab_img_a= dst_Lab_img_a/255.0*(np.max(src_Lab_img_a)-np.min(src_Lab_img_a))+np.min(src_Lab_img_a)

dst_Lab_img_b = imresize(src_Lab_img_b,3.0,'bicubic')
print np.max(src_Lab_img_b),np.min(src_Lab_img_b)
dst_Lab_img_b= dst_Lab_img_b/255.0*(np.max(src_Lab_img_b)-np.min(src_Lab_img_b))+np.min(src_Lab_img_b)

img_lab = np.zeros((dst_Lab_img_L.shape[0],dst_Lab_img_L.shape[1],3))
img_lab[:,:,0] = dst_Lab_img_L
img_lab[:,:,1] = dst_Lab_img_a
img_lab[:,:,2] = dst_Lab_img_b


img = color.lab2rgb(img_lab)
print np.max(img),np.min(img)
plt.imshow(img)

plt.show()
