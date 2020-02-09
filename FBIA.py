import region_smoothing
import etf

img = cv2.imread("images/baboon.jpg",0)
height, width = img.shape

etf(img, 3, 3, True)