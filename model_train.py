import tensorflow as tf
from tf.keras.preprocessing.image import ImageDataGenerator
from tf.keras.models import Sequential
from tf.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Caminho das pastas com as imagens
train_dir = 'fire_dataset/'

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
    target_size=(128, 128),  # Redimensionamento
    batch_size=32,
    class_mode='binary',  # 0 para "non_fire", 1 para "fire"
    subset='training'
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
