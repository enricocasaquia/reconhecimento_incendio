from numpy.typing import ArrayLike
import matplotlib.axes as Figure
import matplotlib.pyplot as plt

class Graph:
    _render: Figure

    def __init__(self, xAxisLabel: str, yAxisLabel: str):
        """
        Inicializa um gráfico com rótulos para os eixos x e y.
        
        Parâmetros
        ----------
        xAxisLabel : str
            Rótulo do eixo X.
        yAxisLabel : str
            Rótulo do eixo Y.
        """

        self._render = plt.subplots()[1]
        self._render.set_xlabel(xAxisLabel)
        self._render.set_ylabel(yAxisLabel)

    def create_line(self, label: str, line: ArrayLike) -> None:
        """
        Cria uma linha no gráfico.

        Parâmetros
        ----------
        label : str
            Nome da linha, usado para legenda.
        line : ArrayLike
            Dados da linha.
        """
        if not isinstance(line, (list, tuple, ArrayLike)):
            raise ValueError("O parâmetro 'line' deve ser ArrayLike para correta renderização.")

        self._render.plot(line, label=label)

    def show(self, title: str = "Gráfico") -> None:
        """
        Exibe o gráfico com o título especificado.
        
        Parâmetros
        ----------
        title : str
            Título do gráfico (padrão é "Gráfico").
        """
        self._render.legend()
        self._render.set_title(title)
        plt.show()
