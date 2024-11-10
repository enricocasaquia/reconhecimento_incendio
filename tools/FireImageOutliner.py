from typing import Sequence
import cv2
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from cv2.typing import MatLike

class FireImageOutliner:
    """
    Classe que representa uma imagem com possibilidade de haver fogo.
    Utiliza filtros e processamento de imagem para identificar áreas que
    podem conter fogo, desenhando contornos nessas regiões e permitindo
    verificação de presença deste (ou não) na imagem.

    Não é uma verificação muito confiável, já que apenas leva em conta
    um campo específico de cores que podem, ou não, significar fogo.

    Atributos
    ---------
    _image : MatLike
        A imagem original, com os contornos desenhados em prováveis regiões
        com fogo.
    _has_fire : bool
        Indicador de presença de fogo na imagem, `True` se contornos foram
        encontrados, caso contrário `False`.
    _LOWER_FIRE_COLOR : ndarray
        Limite inferior para detecção de cor de fogo em HSV.
    _UPPER_FIRE_COLOR : ndarray
        Limite superior para detecção de cor de fogo em HSV.
    """

    _image: MatLike
    _has_fire: bool = False

    _LOWER_FIRE_COLOR: ndarray = np.array([0, 100, 150])
    _UPPER_FIRE_COLOR: ndarray = np.array([15, 245, 255])

    def __init__(self, file_pathname: str) -> None:
        """
        Inicializa uma instância de `FireImageOutliner` carregando uma imagem e aplicando
        filtros e detecção de contornos para marcar áreas de fogo.

        Parâmetros
        ----------
        file_pathname : str
            O caminho do arquivo de imagem a ser carregado.

        Exceções
        --------
        FileNotFoundError
            Se a imagem não for encontrada ou não puder ser carregada.
        """

        the_image: MatLike = cv2.imread(file_pathname)
        if the_image is None:
            raise FileNotFoundError(f"Não foi possível carregar a imagem do caminho: {file_pathname}")

        kernel: ndarray = np.ones((10, 10), np.uint8)

        filtered_image: MatLike = self._apply_image_filters(the_image)
        fire_mask: MatLike = self._create_fire_mask(filtered_image, kernel)

        self._image = self._draw_contours(fire_mask, the_image)

    def show(self, window_title: str = "Outlines na Imagem"):
        """
        Exibe a imagem processada com contornos desenhados em regiões com fogo, 
        se houver. Utiliza uma janela do OpenCV para mostrar o resultado.

        Parâmetros
        ----------
        window_title : str
            Título da janela de exibição da imagem (padrão é "Outlines na Imagem").

        Exceções
        --------
        RuntimeError
            Se houver falha ao tentar exibir a imagem.
        """

        try:
            cv2.imshow(window_title, self._image)
            cv2.waitKey(0)
        except cv2.error as e:
            raise RuntimeError(f"Falha ao exibir a imagem: {e}")
        finally:
            cv2.destroyAllWindows()

    def save(self, file_name: str):
        """
        Salva a imagem processada em um arquivo.

        Parâmetros
        ----------
        file_name : str
            Nome do arquivo onde a imagem será salva.
        
        Exceções
        --------
        IOError
            Se houver falha ao salvar a imagem.
        """

        try:
            cv2.imwrite(file_name, self._image)
        except cv2.error as e:
            raise IOError(f"Erro ao salvar a imagem: {e}")

    def check_if_has_fire(self) -> bool:
        """
        Retorna se a imagem contém fogo, com base na detecção de contornos.

        Retorna
        -------
        bool
            `True` se a imagem contém fogo (ou seja, contornos foram detectados),
            `False` caso contrário.
        """

        return self._has_fire

    def _apply_image_filters(self, image: MatLike) -> MatLike:
        """
        Aplica filtros de pré-processamento à imagem, convertendo-a para RGB,
        aplicando desfoque gaussiano e convertendo para o espaço de cores HSV.

        Parâmetros
        ----------
        image : MatLike
            A imagem original a ser filtrada.

        Retorna
        -------
        MatLike
            A imagem filtrada e convertida para o espaço de cores HSV.
        """
    
        filtered_image: MatLike = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        filtered_image = cv2.GaussianBlur(filtered_image, (5, 5), 0)
        filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_RGB2HSV)

        return filtered_image

    def _create_fire_mask(self, filtered_image: MatLike, kernel: ndarray) -> MatLike:
        """
        Cria uma máscara para detectar regiões que correspondem à cor de fogo.
        Aplica transformações morfológicas para reduzir ruído.

        Parâmetros
        ----------
        filtered_image : MatLike
            A imagem filtrada no espaço de cores HSV.
        kernel : ndarray
            Kernel para operações morfológicas de abertura e fechamento.

        Retorna
        -------
        MatLike
            Máscara binária com regiões de fogo destacadas.
        """

        fire_mask: MatLike = cv2.inRange(filtered_image, self._LOWER_FIRE_COLOR, self._UPPER_FIRE_COLOR)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)

        return fire_mask

    def _draw_contours(self, fire_mask: MatLike, image: MatLike) -> MatLike:
        """
        Desenha contornos em áreas da imagem que contêm fogo, com base na máscara
        gerada. Atualiza o atributo `_has_fire` se contornos forem encontrados.

        Parâmetros
        ----------
        fire_mask : MatLike
            Máscara binária que identifica as regiões de fogo na imagem.
        image : MatLike
            A imagem original onde os contornos serão desenhados.

        Retorna
        -------
        MatLike
            A imagem com contornos desenhados nas áreas de fogo.
        """

        contours: Sequence[MatLike] = cv2.findContours(fire_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        if contours:
            self._has_fire = True

            for contour in contours:
                image = cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        return image
