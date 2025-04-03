import os
import tensorflow as tf

model_path = os.path.expanduser('~/Downloads/cv-course-models/vgg16-pretrained-base')
print(f"Model path: {model_path}")

loaded_model = tf.saved_model.load(model_path)
infer = loaded_model.signatures["serving_default"]

# Ahora puedes usar 'infer' para hacer inferencia.