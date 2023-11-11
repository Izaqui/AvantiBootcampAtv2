import cv2

from PIL import Image, ImageFilter 
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np

imagem = cv2.imread('img1.jpg')
heigth, width, channel = imagem.shape
#Original
R, G, B = imagem[:, :, 0], imagem[:, :, 1], imagem[:, :, 2]
imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
plt.imshow(imgGray, cmap="gray")
plt.show()

#cv2.imshow('Original', imagem)
cv2.waitKey()
gray_img = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

mask = np.ones([3, 3], dtype = int)
mask = mask / 9

newImg = np.zeros([heigth, width])

for i in range(1, heigth-1):
    for j in range(1, width-1):
        temp = imagem[i-1, j-1]*mask[0, 0]+imagem[i-1, j]*mask[0, 1]+imagem[i-1, j + 1]*mask[0, 2]+imagem[i, j-1]*mask[1, 0]+ imagem[i, j]*mask[1, 1]+imagem[i, j + 1]*mask[1, 2]+imagem[i + 1, j-1]*mask[2, 0]+imagem[i + 1, j]*mask[2, 1]+imagem[i + 1, j + 1]*mask[2, 2]
        newImg[i, j] = temp

newImg = newImg.astype(np.uint8)
#cv2.imwrite('Filtro de Media', newImg)   
R, G, B = newImg[:, :, 0], newImg[:, :, 1], newImg[:, :, 2]
newImg = 0.2989 * R + 0.5870 * G + 0.1140 * B
plt.imshow(newImg, cmap="gray")
plt.show()

#mediana
medianaImag = cv2.medianBlur(imagem, 5)
R, G, B = medianaImag[:, :, 0], medianaImag[:, :, 1], medianaImag[:, :, 2]
imgGrayMediana = 0.2989 * R + 0.5870 * G + 0.1140 * B
plt.imshow(imgGrayMediana, cmap="gray")
plt.show()

#Gaussiana
imgGlauci = cv2.GaussianBlur(imagem, (7, 5), 0)
R, G, B = imgGlauci[:, :, 0], imgGlauci[:, :, 1], imgGlauci[:, :, 2]
imgGrayGaussian = 0.2989 * R + 0.5870 * G + 0.1140 * B
plt.imshow(imgGrayGaussian, cmap="gray")
plt.show()





##cv2.imshow('Glauciano', imgGlauci)

#cv2.waitKey()
#cv2.destroyAllWindows()

