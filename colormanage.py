# coding:utf-8
__author__ = 'liangz14'
import numpy as np
from skimage import data
import matplotlib.pyplot as plt

def rgb2ycbcr(img):
    YCbCr = np.zeros(img.shape,dtype='float64')
    YCbCr[:,:,0] = 0.299*img[:,:,0]+0.587*img[:,:,1]+0.114*img[:,:,2]
    YCbCr[:,:,1] = 128 - 0.168736*img[:,:,0] - 0.331264*img[:,:,1]+ 0.5*img[:,:,2]
    YCbCr[:,:,2] = 128 + 0.5*img[:,:,0] - 0.418688*img[:,:,1] - 0.081312*img[:,:,2]
    return YCbCr

def ycbcr2rgb(YCbCr):
    img = np.zeros(YCbCr.shape,dtype='float64')
    img[:,:,0] = YCbCr[:,:,0] + 1.402 * (YCbCr[:,:,2] - 128 )
    img[:,:,1] = YCbCr[:,:,0] - 0.34414 * (YCbCr[:,:,1] - 128) - 0.71414 *(YCbCr[:,:,2]-128)
    img[:,:,2] = YCbCr[:,:,0] + 1.772 * (YCbCr[:,:,1] - 128)
    return img


def test_main():
    src_rgb_img = data.lena()

    src_YCrCb_img = rgb2ycbcr(src_rgb_img)
    p = ycbcr2rgb(src_YCrCb_img)

    c =np.asarray(p,dtype='uint8')
    plt.imshow(c)
    print src_rgb_img.dtype,src_rgb_img.shape,np.max(src_rgb_img),np.min(src_rgb_img)
    print c.dtype,c.shape,np.mean((c-src_rgb_img)**2),np.max(c),np.min(c)

    plt.show()

#test_main()