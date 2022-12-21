import numpy as np
import cv2

from matplotlib import pyplot as plt


xml_haarcascade = "haarcascade_frontalface_alt2.xml"                         # Arquivo classificador

face_classifier = cv2.CascadeClassifier(xml_haarcascade)                     # Carrega o clissificador.


# Inicia camera:
capture = cv2.VideoCapture(0)

# Define o tamanho da imagem:
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 340)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 350)


while not cv2.waitKey(20) & 0xFF == ord("q"):
    ret, frame_color = capture.read()

    gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)                    # Tela em cinza.
    faces = face_classifier.detectMultiScale(gray)                          # Detecta os rostos.

    for x, y, w, h in faces:
        cv2.rectangle(frame_color, (x,y), (x + w, y + h),(0,0,255), 2)


    cv2.imshow('color', frame_color)
    cv2.imshow('gray', gray)
