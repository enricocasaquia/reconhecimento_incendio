# reconhecimento_incendio

Projeto de rede neural convolucional (CNN) para detecção de focos de incêndio em imagens, com o objetivo de identificar incêndios em áreas florestais e ambientes sensíveis.

## Descrição

Este projeto utiliza uma rede neural convolucional (CNN) para analisar imagens e identificar se há presença de fogo. A rede foi treinada com um conjunto de dados de imagens contendo exemplos de áreas com e sem fogo, e o modelo resultante pode ser usado para detectar focos de incêndio em imagens de entrada. O projeto é dividido em módulos que lidam com a detecção de incêndio, a exibição de outlines (contornos) das áreas detectadas, e a visualização dos resultados.

A estrutura do código foi desenvolvida para ser modular, facilitando a manutenção e a possibilidade de futuras expansões, como o ajuste do modelo ou a inclusão de novas técnicas de processamento de imagem.

## Estrutura do Projeto

```
├── models                   # Modelos treinados salvos no formato .keras
├── samples                  # Diretório com imagens de exemplo para testes
│   ├── fire                 # Imagens contendo fogo
│   └── not_fire             # Imagens sem fogo
├── tools                    # Ferramentas para detecção e análise de imagens
│   ├── FireDetection.py     # Classe principal para detecção de incêndio
│   ├── FireImageOutliner.py # Classe para exibir contornos (outlines) de incêndio nas imagens
│   └── Graph.py             # Classe de visualização e gráficos de performance do modelo
├── .gitignore
├── LICENSE                  # Licença do projeto
├── main.py                  # Script principal para execução do modelo
├── README.md                # Descrição do projeto
└── requirements.txt         # Dependências do projeto
```

## Instalação e Uso

### Pré-requisitos

-   Python 3.8 ou superior;
-   TensorFlow e Keras
-   Outras bibliotecas listadas em `requirements.txt`

### Passo a Passo de Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu_usuario/reconhecimento_incendio.git
cd reconhecimento_incendio
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

### Uso

1. Executando o script principal:

```bash
python main.py
```

2. Treinamento do modelo:

No menu inicial, escolha a opção para "criar" um novo modelo, e forneça:

-   O caminho para o dataset
-   O número de épocas (quantas vezes o modelo passará pelo dataset durante o treinamento)

3. Carregando um Modelo Existente

Escolha a opção "carregar" para carregar um modelo previamente treinado e iniciar a detecção em novas imagens.

4. Testando Imagens ou Diretórios

Após carregar ou treinar o modelo, o programa permite que você teste uma imagem específica ou todas as imagens de um diretório. Esse processo analisa cada imagem e indica se há fogo presente.

5. Exibição de Outlines

Você pode optar por visualizar outlines (contornos) das áreas de fogo detectadas em uma imagem específica, usando a classe FireImageOutliner.

## Bibliografia e Recursos

Para desenvolvimento e treinamento do modelo, foram usados diversos recursos e datasets:

### Datasets

A nossa própria dataset, criada usando as datasets abaixo:

-   https://www.kaggle.com/datasets/dani215/fire-dataset/

-   https://www.kaggle.com/datasets/cristiancristancho/forest-fire-image-dataset
-   https://www.kaggle.com/datasets/kutaykutlu/forest-fire
-   https://www.kaggle.com/datasets/atulyakumar98/test-dataset
-   https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images
-   https://www.kaggle.com/datasets/muratkokludataset/acoustic-extinguisher-fire-dataset
-   https://www.kaggle.com/datasets/akhiljethwa/forest-vs-desert
-   https://www.kaggle.com/datasets/informaticteens/yellow-and-read-autumn-color-grapeleafs

### Documentação

-   https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
-   https://github.com/keras-team/keras/blob/v3.3.3/keras/src/legacy/preprocessing/image.py#L949-L1547
-   https://www.tensorflow.org/api_docs/python/tf/keras/Model
-   https://github.com/keras-team/keras/blob/v3.3.3/keras/src/backend/tensorflow/trainer.py#L236-L369
-   https://www.tensorflow.org/api_docs/python/tf/keras/utils/load_img
-   https://pillow.readthedocs.io/en/stable/reference/Image.html

### Tutoriais e Artigos

-   https://medium.com/@teamcode20233/how-to-import-class-from-another-file-in-python-179c3a4092a7
-   https://www.geeksforgeeks.org/python-opencv-destroyallwindows-function/

Licença
Este projeto é distribuído sob a licença MIT - veja o arquivo `LICENSE` para mais detalhes.
