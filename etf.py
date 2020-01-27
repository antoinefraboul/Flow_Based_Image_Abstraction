import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from PIL import Image

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
    return (g_y - g_x + 1)/2

def computeWd(x, y):
    if(np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0):
        # print("norm is 0")
        return 0
    else: 
        # print(abs(np.dot(x/np.linalg.norm(x),y/np.linalg.norm(y))))
        return abs(np.dot(x/np.linalg.norm(x),y/np.linalg.norm(y)))

# EFT function
def etf(img, kernel_size, img_grad, img_t):

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    img_t_res = [[[0,0] for x in range(width)] for y in range(height)] # t(^x)
    
    for i in range(len(img_t_res)):
        for j in range(len(img_t_res[0])):
            img_t_res[i][j] = etfIter(i, j, kernel_size, img_t, img_grad)
            # print("sum: ",img_t_res[i][j])
    return img_t_res

# One iteration of ETF
def etfIter(x_a, x_b, kernel, img_t, img_grad):
    
    sum = [0,0]
    k = kernel**2
    
    height, width = len(img_t), len(img_t[0])

    for y_a in range(x_a-kernel//2, x_a+kernel//2):
        for y_b in range(x_b-kernel//2, x_b+kernel//2):
            # Gestion des bords
            if(y_a<0 or y_b<0 or y_a>=height or y_b>=width): 
                continue
            res = [0,0]
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
            sum[0] += (img_t[y_a][y_b][0] * weigths) / k
            sum[1] += (img_t[y_a][y_b][1] * weigths) / k
    
    return sum

def displayImgT(img_t, text, block):
    height, width = img.shape
    img_res_norm = np.zeros((height,width), np.float64)

    for i in range(height):
        for j in range(width):
            img_res_norm[i][j] = math.sqrt( (img_t[i][j][0] **2 ) + (img_t[i][j][1] **2 ) ) 

    max_val = np.max(img_res_norm)
    print(text+" size :", len(img_res_norm), " - ", len(img_res_norm[0]))
    # print(img_res_norm)
    #cv2.imwrite("ETF_normalized.jpg", img_res_norm/max_val * 255)
    cv2.imshow(text, img_res_norm/max_val)
    if block : cv2.waitKey(0)

# Main
img = cv2.imread("images/baboon.jpg",0)
height, width = img.shape

# Solbel filter to get the gradient
sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

img_grad = np.hypot(sobel_x, sobel_y) # Take the magnitude of the gradient
img_grad = cv2.normalize(img_grad.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX) # Normalize the gradient magnitude

img_t = [[[0,0] for x in range(width)] for y in range(height)] # t(y)
for i in range(len(img)):
        for j in range(len(img[i])):
            img_t[i][j] = [sobel_y[i][j], -sobel_x[i][j]]

displayImgT(img_t, "Img t before", False)

kernel_size = 3

for i in range(0,3):
    img_t = etf(img, kernel_size, img_grad, img_t)

img_res = img_t

print("ETF not normalized:", len(img_res), " - ", len(img_res[0]))
# cv2.imshow("ETF not normalized",img_res)

# Display
displayImgT(img_res, "Img res", True)
