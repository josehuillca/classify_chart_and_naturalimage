import tensorflow as tf
import visualkeras
from PIL import ImageFont

# 1. Define a simple CNN model (example using Keras Sequential API)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(3, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512*3*3, input_shape=(2,), activation='softmax')
])

# 2. Generate and display the visualization
print("Generating CNN visualization...")

# You can customize the look using different styles and colors
visualkeras.layered_view(model, legend=True).show()

# To save to a file instead:
# visualkeras.layered_view(model, to_file='cnn_architecture.png', legend=True)
print("Visualization complete.")