import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('samples/incendio1.png')

imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
imagem_suave = cv2.GaussianBlur(imagem_rgb, (5, 5), 0)

bordas = cv2.Canny(imagem_suave, 100, 200)

plt.imshow(bordas)
plt.show()