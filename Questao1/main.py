import cv2
import numpy as np


imagem = cv2.imread('img.jpg')
heigth, width, channel = imagem.shape

#cv2.imshow('Original', imagem)
cv2.waitKey()
gray_img = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

cv2.imshow('Tom de Cinza', gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f'Altura: {heigth}')
print(f'Largura: {width}')
print(f'Channel:{channel}')

print(f'Tom de cinza medio: {gray_img.mean()}')
print(f'Tom de cinza maximo: {gray_img.max()}')
print(f'Tom de cinza minimo: {gray_img.min()}')