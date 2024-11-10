import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, History

from tools.Graph import Graph

from PIL import Image
from tensorflow.keras import Model

class FireDetection:
    """
    Classe para detecção de fogo em imagens utilizando uma rede neural convolucional (CNN) com Keras e TensorFlow.

    Atributos
    ---------
    model : Model
        O modelo de rede neural para detecção de fogo.
    history : History
        Histórico de treinamento do modelo.
    image_size : tuple[int, int]
        Tamanho das imagens de entrada (largura, altura).
    batch_size : int
        Tamanho do lote usado durante o treinamento e validação.
    datagenConfig : dict[str, any]
        Configurações para o gerador de dados, incluindo transformação e divisão de validação.
    """

    model: Model = None
    history: History = None
    image_size: tuple[int, int]
    batch_size: int
    datagenConfig: dict[str, any]

    def __init__(self) -> None:
        """
        Inicializa a classe FireDetection, definindo o tamanho das imagens, batch size e configurações do gerador de dados.
        """

        self.model: Model = None
        self.history: History = None
        self.image_size: tuple[int, int] = (128, 128)
        self.batch_size: int = 32
        self.datagenConfig: dict[str, any] = {
            "rescale": 1.0/255,
            "validation_split": 0.2,
            "horizontal_flip": True,
            "rotation_range": 15,
            "brightness_range": [0.7, 1.3],
            "zoom_range": [0.9, 1.1]
        }

    def initialize_model(self, model_pathname: str, training_dataset_pathname: str, epoch_quantity: int) -> None:
        """
        Inicializa, treina e salva o modelo de detecção de fogo.

        Parâmetros
        ----------
        model_pathname : str
            Caminho para salvar o modelo após o treinamento.
        training_dataset_pathname : str
            Caminho para o diretório do dataset de treinamento.
        epoch_quantity : int
            Número de épocas para o treinamento.
        """

        self._create_and_configure_model()
        self._train_model(model_pathname, training_dataset_pathname, epoch_quantity)

    def load_model(self, model_pathname: str) -> None:
        """
        Carrega um modelo previamente treinado de um arquivo.

        Parâmetros
        ----------
        model_pathname : str
            Caminho do arquivo do modelo a ser carregado.
        """

        self.model = tf.keras.models.load_model(self._ensure_keras_extension(model_pathname))

    def get_model_summary(self) -> None:
        """
        Exibe o resumo do modelo, incluindo a arquitetura e número de parâmetros.
        """

        if self.model:
            self.model.summary()
        else:
            print("Modelo não foi carregado ou treinado ainda para poder exibir seu resumo.")

    def test_if_image_has_fire(self, img_path: str) -> float:
        """
        Prediz a probabilidade de uma imagem conter fogo.

        Parâmetros
        ----------
        img_path : str
            Caminho para a imagem a ser testada.

        Retorna
        -------
        float
            Percentual de confiança de que a imagem contém fogo.
        """

        if self.model is None:
            print("Modelo não foi carregado. Por favor, carregue ou treine o modelo primeiro.")
            return 0.0

        image: Image = tf.keras.utils.load_img(img_path, target_size=(128, 128))
        image_as_array: np.ndarray = tf.keras.utils.img_to_array(image) / 255.0
        image_as_array: np.ndarray = np.expand_dims(image_as_array, axis=0)

        prediction: float = (1 - self.model.predict(image_as_array)[0][0]) * 100

        print(f"Imagem {img_path} teve como resultado: {prediction:.2f}%")

        return prediction

    def _ensure_keras_extension(self, file_name: str) -> str:
        """
        Garante que o nome do arquivo tenha a extensão '.keras'.

        Parâmetros
        ----------
        file_name : str
            Nome do arquivo.

        Retorna
        -------
        str
            Nome do arquivo com a extensão '.keras'.
        """

        return f"{file_name}.keras" if not file_name.endswith(".keras") else file_name

    def _create_and_configure_model(self) -> Model:
        """
        Cria e configura a arquitetura do modelo CNN para detecção de fogo.

        Retorna
        -------
        Model
            O modelo CNN configurado.
        """

        model: Model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model
    
    def _train_model(self, model_pathname: str, training_dataset_pathname: str, epoch_quantity: int) -> None:
        """
        Treina o modelo usando o conjunto de dados de treinamento e salva o modelo após o treinamento.

        Parâmetros
        ----------
        model_pathname : str
            Caminho para salvar o modelo treinado.
        training_dataset_pathname : str
            Caminho para o conjunto de dados de treinamento.
        epoch_quantity : int
            Número de épocas para o treinamento.
        """

        datagen: ImageDataGenerator = self._get_data_generator()

        train_data: DirectoryIterator = self._get_training_data_iterators(datagen, training_dataset_pathname)
        validation_data: DirectoryIterator = self._get_validation_data_iterators(datagen, training_dataset_pathname)

        self.history: History = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epoch_quantity,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
        ).history

        self.model.save(self._ensure_keras_extension(model_pathname))

        model_evaluation: str = self._get_model_evaluation(validation_data)

        print(model_evaluation)
        self._plot_training_history()

    def _get_data_generator(self) -> ImageDataGenerator:
        """
        Cria o gerador de dados com as configurações definidas.

        Retorna
        -------
        ImageDataGenerator
            O gerador de dados configurado.
        """

        datagen: ImageDataGenerator = ImageDataGenerator(**self.datagenConfig)

        return datagen
    
    def _get_training_data_iterators(self, datagen: ImageDataGenerator, dataset_path: str) -> DirectoryIterator:
        """
        Obtém o iterador para os dados de treinamento.

        Parâmetros
        ----------
        datagen : ImageDataGenerator
            O gerador de dados configurado.
        dataset_path : str
            Caminho para o diretório do dataset.

        Retorna
        -------
        DirectoryIterator
            O iterador de dados de treinamento.
        """

        training_data: DirectoryIterator = datagen.flow_from_directory(
            dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='training',
            classes={'fire': 1, 'not_fire': 0}
        )

        return training_data

    def _get_validation_data_iterators(self, datagen: ImageDataGenerator, dataset_path: str) -> DirectoryIterator:
        """
        Obtém o iterador para os dados de validação.

        Parâmetros
        ----------
        datagen : ImageDataGenerator
            O gerador de dados configurado.
        dataset_path : str
            Caminho para o diretório do dataset.

        Retorna
        -------
        DirectoryIterator
            O iterador de dados de validação.
        """

        validation_data: DirectoryIterator = datagen.flow_from_directory(
            dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='binary',
            subset='validation',
            classes={'fire': 1, 'not_fire': 0}
        )

        return validation_data

    def _get_model_evaluation(self, validation_data: DirectoryIterator) -> str:
        """
        Avalia o modelo com os dados de validação e retorna a acurácia e perda.

        Parâmetros
        ----------
        validation_data : DirectoryIterator
            Dados de validação para avaliar o modelo.

        Retorna
        -------
        str
            Acurácia e perda do modelo em formato de string.
        """

        val_loss, val_accuracy = self.model.evaluate(validation_data)

        return (
        f"""
Acurácia na validação: {(val_accuracy * 100):.2f}%
Perda na validação: {(val_loss * 100):.2f}%
        """)

    def _plot_training_history(self) -> None:
        """
        Plota o histórico de acurácia e perda do treinamento e da validação.
        """

        if not self.history:
            print("Histórico de treinamento vazio.")

            return

        graph: Graph = Graph("Épocas", "Métricas")
        graph.create_line('Acurácia de Treinamento', self.history["accuracy"])
        graph.create_line('Acurácia de Validação', self.history["val_accuracy"])
        graph.show("Gráfico de Precisão")

        graph: Graph = Graph("Épocas", "Perda")
        graph.create_line('Perda de Treinamento', self.history["loss"])
        graph.create_line('Perda de Validação', self.history["val_loss"])
        graph.show("Gráfico de Perda")
