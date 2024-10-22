import cv2
import numpy as np
import matplotlib.pyplot as plt

imagem = cv2.imread('samples/incendio2.jpg')
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB) #IMAGEM RGB ORIGINAL
imagem_suave = cv2.GaussianBlur(imagem_rgb, (5, 5), 0) #IMAGEM COM MENOS RUIDO
# bordas = cv2.Canny(imagem_suave, 10, 50)

imagem_hsv = cv2.cvtColor(imagem_suave, cv2.COLOR_RGB2HSV) #IMAGEM HSV PARA ANALISE

lower_fire = np.array([26, 50, 150])
upper_fire = np.array([35, 240, 255])
mascara_fogo = cv2.inRange(imagem_hsv, lower_fire, upper_fire)
result = cv2.bitwise_and(imagem_rgb, imagem_rgb, mask=mascara_fogo)

contours, _ = cv2.findContours(mascara_fogo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 500:  # Definir um limite de área mínima
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(imagem_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Imagem Original', imagem_rgb)
cv2.imshow('Focos de incêndio', result)

cv2.waitKey(0)
cv2.destroyAllWindows()