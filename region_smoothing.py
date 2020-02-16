from matplotlib import pyplot as mp
import numpy as np
import cv2
from sklearn import preprocessing

sigma_e = 2
r_e = 50
sigma_g = 2
r_g = 10

alpha = 2 * sigma_e
beta = 2 * sigma_g

def kernel_distance(x, y, dist) :
    return max(abs(x[0]- y[0]), abs(x[1]- y[1])) < dist

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def spatialWeight(sigma, s) :
    return gaussian(s,0,sigma)
    
def similarityWeight(x,y, sigma) :
    val = abs(x-y)
    return gaussian(val,0,sigma)

def Ce_bis(img, etf_x, etf_y) :
    size = img.shape
    res = np.zeros(size)
    kernel_lenght = 1
    img[0,0] = 0

    for i in range(size[0]) :
        for j in range(size[1]) :
            ce = 0
            norm = 0
            for  k in range(-kernel_lenght, kernel_lenght+ 1) :
                vec = [( etf_y[i,j] , -etf_x[i,j])]
                normalized_lx = preprocessing.normalize(vec, norm='l1')
                lx = (i, j) + k * normalized_lx
                t =  (i, j) + k * normalized_lx 
                if  ((lx[0] >= 0) and (lx[1] >= 0) and (lx[0] <= (size[0]-1 ))and (lx[1] <= (size[0]-1)) ):
                    norm_val = spatialWeight( img[  lx[0] , lx[1]] ) * similarityWeight( img[ i , j ] ,  img [ lx[0] , lx[1] ] , r_g )
                    ce +=  img[ lx[0] ,  lx[1] ] * norm_val
            norm += norm_val

            res[i,j] = ce / norm

def Cv(img, etf_x, etf_y) :
    height, width, __ = img.shape
    res = np.zeros(height, width)

    for i in range(height) :
        for j in range(width) :
            norm = 0
            val = 0

            x = y = y_neg = (i,j)
            norm_mode = True

            while (norm_mode) :
                mode_pos = mode_neg = False

                y = (round(y[0] + etf_x[x[0],x[1]]), round(y[0] + etf_y[x[0],x[1]]))
                if(kernel_distance(x,y,alpha)) :
                    norm += spatialWeight(img[i,j], sigma_e) * similarityWeight(img[i,j], img[y[0],y[1]], sigma_g)
                    mode_pos = True

                y_neg = (round(y_neg[0] - etf_x[x[0],x[1]]), round(y_neg[0] - etf_y[x[0],x[1]]))
                if(kernel_distance(x,y_neg,alpha)) :
                    norm += spatialWeight(img[i,j], sigma_e) * similarityWeight(img[i,j], img[y_neg[0],y_neg[1]], sigma_g)
                    mode_pos = True

                norm_mode = mode_neg | mode_pos

            y = y_neg = (i,j)
            in_kernel = True

            while (in_kernel) :
                kernel_pos = kernel_neg = False

                y = (round(etf_x[y[0],y[1]]), round(etf_y[y[0],y[1]]))
                if(kernel_distance(x,y,alpha)) :
                    val_unnorm= img[y[0], y[1]] * spatialWeight(img[i,j], sigma_e) * similarityWeight(img[i,j], img[y[0], y[1]], sigma_g) 
                    val = val_unnorm / norm
                    kernel_pos = True

                y_neg = (round(-etf_x[y_neg[0],y_neg[1]]), round(-etf_y[y_neg[0],y_neg[1]]))
                if(kernel_distance(x,y_neg,alpha)) :
                    val_unnorm = img[y_neg[0], y_neg[1]] * spatialWeight(img[i,j], sigma_e) * similarityWeight(img[i,j], img[y_neg[0], y_neg[1]], sigma_g) 
                    val = val_unnorm / norm
                    kernel_neg = True

                in_kernel = kernel_pos | kernel_neg
                
            res[i,j] = val
            
    return res

def filtering(img, etf_x, etf_y) :

    for __ in range(5) :
        img = Ce_bis(img, etf_x, etf_y)
        img = Cv(img, etf_x, etf_y)

    return img