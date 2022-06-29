'''
Autor: www.github.com/JuanBindez
Descrição: Detecta Rostos Em Imagens
'''

import os
import time
import cv2

# carrega modelo
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_defalt.xml')

# Carrega imagem
img = cv2.imread('imagem.jpg')


# Converte para escala de cinzaclear
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Detecta rostos
faces = face_cascade.detectMultiScale(gray, 1.1, 4)


# Desenha retangulos no rosto capturado
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


# Mostra o resultado
cv2.imshow('img', img)
cv2.waitKey()
