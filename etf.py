import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from licpy.lic import runlic
from licpy.plot import grey_save

def computePhi(x, y):
    if np.dot(x, y) > 0:
        return 1
    else: return -1

def computeWs(a, b, kernel_size):
    norm = np.linalg.norm(a - b)
    # print("a", a)
    # print("b", b)
    # print('Norm :', norm)
    if norm < kernel_size:
        return 1
    else: return 0

def computeWm(g_x, g_y):
    return (1 + np.tanh(g_y - g_x))/2

def computeWd(x, y):
    if(np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0):
        # print("norm is 0")
        return 0
    else: 
        # print(abs(np.dot(x/np.linalg.norm(x),y/np.linalg.norm(y))))
        return abs(np.dot(x/np.linalg.norm(x),y/np.linalg.norm(y)))

# EFT function
def etfIter(img, kernel_size, img_grad, img_t):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    img_t_res = [[[0,0] for x in range(width)] for y in range(height)] # t(^x)
    
    for i in range(len(img_t_res)):
        for j in range(len(img_t_res[0])):
            img_t_res[i][j] = etfKernel(i, j, kernel_size, img_t, img_grad)
            # print("sum: ",img_t_res[i][j])
    return img_t_res

# One iteration of ETF
def etfKernel(x_a, x_b, kernel, img_t, img_grad):
    
    sum = (0,0)
    # k = kernel**2
    k = 0

    height, width = len(img_t), len(img_t[0])

    for y_a in range(x_a-kernel//2, x_a+kernel//2 + 1):
        for y_b in range(x_b-kernel//2, x_b+kernel//2 + 1):
            # Gestion des bords
            if(y_a<0 or y_b<0 or y_a>=height or y_b>=width): 
                continue
            phi = computePhi(img_t[x_a][x_b], img_t[y_a][y_b])
            
            a = np.array([x_a, x_b])
            b = np.array([y_a, y_b])
            ws = computeWs(a, b, kernel)
            wm = computeWm(img_grad[x_a][x_b], img_grad[y_a][y_b])
            wd = computeWd(img_t[x_a][x_b], img_t[y_a][y_b])
            
            # print("phi :", phi)
            # print("ws :", ws)
            # print("wm :", wm)
            # print("wd :", wd)

            weigths = phi * ws * wm * wd
            sum += (img_t[y_a][y_b] * weigths)# / k
            #sum[1] += (img_t[y_a][y_b][1] * weigths)# / k

            k += weigths

    #if k !=0 :
     #   sum[0] = sum[0] / k
      #  sum[1] = sum[1] / k
    n = sum
    cv2.normalize(sum, n)
    return sum

def displayImgT(img_t, text):
    height, width = img.shape
    img_res_norm = np.zeros((height,width), np.float32)

    for i in range(height):
        for j in range(width):
            img_res_norm[i][j] = math.sqrt( (img_t[i][j][0] **2 ) + (img_t[i][j][1] **2 ) ) 

    max_val = np.max(img_res_norm)
    print(text+" size :", len(img_res_norm), " - ", len(img_res_norm[0]))
    
    #cv2.imwrite("ETF_normalized.jpg", img_res_norm/max_val * 255)
    cv2.imshow(text, img_res_norm/max_val)

def etf(img_src, nb_iter, kernel_size, display):
    
    height, width = img_src.shape
    size = (height, width, 3)

    src_n = np.zeros(size, dtype = np.float32)
    src_n = cv2.normalize(img_src.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Solbel filter to get the gradient
    sobel_x = cv2.Sobel(src_n,cv2.CV_32FC1,1,0,ksize=5)
    sobel_y = cv2.Sobel(src_n,cv2.CV_32FC1,0,1,ksize=5)

    # img_grad = np.hypot(sobel_x, sobel_y) # Take the magnitude of the gradient
    # img_grad = cv2.normalize(img_grad.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX) # Normalize the gradient magnitude
    img_grad = cv2.sqrt(sobel_x**2.0 + sobel_y**2.0) 
    img_grad = cv2.normalize(img_grad.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)


    img_t = [[[0,0] for x in range(width)] for y in range(height)] # t(y)

    for i in range(len(img)):
        for j in range(len(img[i])):
            img_t[i][j] = [sobel_y[i][j], -sobel_x[i][j]]

    cv2.imshow("img_grad", img_grad)

    for i in range(0,nb_iter):
        img_t = etfIter(img, kernel_size, img_grad,  np.copy(img_t))

    h = len(img_t)
    w = len(img_t[0])
    vx = np.zeros((h,w))
    vy = np.zeros((h,w))
    
    for i in range(0, h):
        for j in range (0, w):
            vx[i][j] = img_t[i][j][0]
            vy[i][j] = img_t[i][j][1]

    tex = runlic(vx, vy, 21)
    grey_save("res/lic.jpg", tex)
    
    if display:
        displayImgT(img_t, "Img res")
        cv2.waitKey(0)

    return img_t

# Main
if __name__ == '__main__':
    img = cv2.imread("images/baboon.jpg", 0)

    etf(img, 3, 3, True)
