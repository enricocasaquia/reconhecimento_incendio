import cv2
import numpy as np
import matplotlib.pyplot as plt


# Leitura e conversão da imagem
imagem = cv2.imread('fire_dataset/fire_images/fire.1.png')
imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)  # Conversão para RGB
imagem_suave = cv2.GaussianBlur(imagem_rgb, (5, 5), 0)  # Aplicação de GaussianBlur para suavizar

# Conversão para HSV
imagem_hsv = cv2.cvtColor(imagem_suave, cv2.COLOR_RGB2HSV)

# Intervalos de cor HSV para detecção de fogo
lower_fire = np.array([0, 100, 150])  # Ajustar o valor conforme necessário
upper_fire = np.array([15, 245, 255])

# Criação da máscara de detecção de fogo
mascara_fogo = cv2.inRange(imagem_hsv, lower_fire, upper_fire)

# Aplicação de operações morfológicas para remover ruídos e preencher áreas
kernel = np.ones((10, 10), np.uint8)
mascara_fogo = cv2.morphologyEx(mascara_fogo, cv2.MORPH_CLOSE, kernel)
mascara_fogo = cv2.morphologyEx(mascara_fogo, cv2.MORPH_OPEN, kernel)

# Resultado da aplicação da máscara na imagem original
result = cv2.bitwise_and(imagem, imagem, mask=mascara_fogo)

# Detecção de contornos
contours, _ = cv2.findContours(mascara_fogo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Desenho de retângulos ao redor dos focos detectados
for contour in contours:
    area = cv2.contourArea(contour)
    if 0 < area < 999999:  # Definindo limites de área mínima e máxima
        # Aproximação do contorno para verificar irregularidade
        perimetro = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimetro, True)

        # Verifica se o contorno é "irregular" (mais de 4 vértices)
        if len(approx) > 4:
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 254, 0), 2)  # Cor verde
            cv2.drawContours(imagem, [contour], -1, (0, 255, 0), 2)

         # Verificação de convexidade e bordas irregulares
        #if len(approx) > 4 and not cv2.isContourConvex(approx):
        #    cv2.drawContours(imagem, [contour], -1, (0, 255, 0), 2)  # Desenha contorno em verde

# Exibição das imagens
cv2.imshow('Imagem Original', imagem)
cv2.imshow('Focos de incendio', result)

# Salvar o resultado final com os focos de incêndio detectados
cv2.imwrite('focos_detectados.jpg', imagem)

cv2.waitKey(0)
cv2.destroyAllWindows()
