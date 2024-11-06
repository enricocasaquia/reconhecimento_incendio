import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

# Caminho das pastas com as imagens
train_dir = 'fire_dataset/'

def load_data():
    # Gerador de imagens com data augmentation
    datagen = ImageDataGenerator(
        rescale=1.0/255,  # Normalização para 0-1
        validation_split=0.2,  # Dividir 20% para validação
        horizontal_flip=True,
        rotation_range=15
    )

    # Carregar as imagens para treinamento e validação
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    
    return train_data, val_data

def build_model():
    # Definição do modelo
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compilação do modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    # Plotando a acurácia
    plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.show()

    # Plotando a perda
    plt.plot(history.history['loss'], label='Perda de Treinamento')
    plt.plot(history.history['val_loss'], label='Perda de Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()
    plt.show()

def evaluate_model(model, val_data):
    val_loss, val_accuracy = model.evaluate(val_data)
    print(f'Acurácia na validação: {val_accuracy:.2f}')
    print(f'Perda na validação: {val_loss:.2f}')

def predict_image(model, img_path):
    # Atualização para `tf.keras.utils` para carregar e pré-processar a imagem
    img = tf.keras.utils.load_img(img_path, target_size=(128, 128))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0] > 0.25:
        print("Fogo detectado na imagem -", prediction)
    else:
        print("Nenhum foco de fogo detectado -", prediction)

def main():
    model = build_model()
    
    # Resumo do modelo
    model.summary()

    train_data, val_data = load_data()
    
    # Treinamento do modelo
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=2
    )
    
    # Plotar o histórico de treinamento
    plot_training_history(history)
    
    # Avaliar o modelo com dados de validação
    evaluate_model(model, val_data)
    
    # Testar uma imagem individual (modifique o caminho para a imagem)
    for i in range(1,10):
        img_path = 'fire_dataset/fire_images/fire.'+str(i)+'.png'
        predict_image(model, img_path)

if __name__ == '__main__':
    main()
