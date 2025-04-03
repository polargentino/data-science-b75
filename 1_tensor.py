# Imports
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# Reproducibilidad
def set_seed(seed=31415):
    """Establece las semillas para la reproducibilidad."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed(31415)

# Configuración de Matplotlib
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")  # Limpiar la salida

# 1. Cargar datos
# Rutas actualizadas según tu sistema de archivos
train_dir = '/home/pol/Downloads/train'
valid_dir = '/home/pol/Downloads/valid'  # Asegúrate de que esta ruta sea correcta

ds_train_ = image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    valid_dir,
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)
print(f"Found {len(ds_train_)*64} files belonging to 2 classes.")
print(f"Found {len(ds_valid_)*64} files belonging to 2 classes.")

# Preprocesamiento de datos
def convert_to_float(image, label):
    """Convierte las imágenes a tipo float32."""
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE  # Usar tf.data.AUTOTUNE directamente
ds_train = (
    ds_train_
    .map(convert_to_float, num_parallel_calls=AUTOTUNE)  # Paralelizar
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float, num_parallel_calls=AUTOTUNE)  # Paralelizar
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# 2. Definir la base preentrenada (VGG16)
# Descargar VGG16 directamente desde Keras si no tienes el archivo
pretrained_base = tf.keras.applications.VGG16(
    weights='imagenet',  # Usar pesos preentrenados de ImageNet
    include_top=False,  # No incluir la capa de clasificación final
    input_shape=(128, 128, 3)  # Ajustar al tamaño de las imágenes
)
pretrained_base.trainable = False

# 3. Adjuntar la cabeza
model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(64, activation='relu'),  # Aumentar las unidades
    layers.Dropout(0.5),  # Agregar Dropout para regularización
    layers.Dense(1, activation='sigmoid'),
])

# 4. Entrenar el modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

# Agregar EarlyStopping para evitar overfitting
early_stopping = keras.callbacks.EarlyStopping(
    patience=5,  # Detener si no hay mejora en 5 epochs
    min_delta=0.001,  # Requerir una mejora mínima
    restore_best_weights=True,
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    verbose=1,  # Mostrar el progreso del entrenamiento
    callbacks=[early_stopping],  # Agregar EarlyStopping
)

# 5. Visualizar el entrenamiento
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
plt.title("Pérdida durante el entrenamiento")
plt.show()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.title("Precisión durante el entrenamiento")
plt.show()
