import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

Image.open('suza.jpg', mode='RGB')

# EFT function
def etf(kernel, image):
    image.convert('LA')
    
    height, width = image.shape
    
    for i in range(0, height):
        for j in range(0, width):
            image[i,j]=etfIter(i, j, kernel)

# One iteration of ETF
def etfIter(i, j, kernel):
    
    sum = 0
    numrows = len(kernel)    # 3 rows in your example
    numcols = len(kernel[0]) # 2 columns in your example
    k = numcols * numrows
    return 1,1

# Normalised gradient magnitude
def normalizedGrad():
    print("coucou2")
    return 0