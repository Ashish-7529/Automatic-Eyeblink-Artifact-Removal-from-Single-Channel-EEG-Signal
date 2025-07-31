# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 23:53:15 2024

@author: EED
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 04:48:18 2024

@author: EED
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr
# Load data
x_train_noisy = np.load('C:/Users/EED/Desktop/Btech 2024/noiseEEG_train.npy')
x_train_clean = np.load('C:/Users/EED/Desktop/Btech 2024/EEG_train.npy')
x_val_noisy = np.load('C:/Users/EED/Desktop/Btech 2024/noiseEEG_val.npy')
x_val_clean = np.load('C:/Users/EED/Desktop/Btech 2024/EEG_val.npy')
x_test_noisy = np.load('C:/Users/EED/Desktop/Btech 2024/noiseEEG_test.npy')
x_test_clean = np.load('C:/Users/EED/Desktop/Btech 2024/EEG_test.npy')

# Define the model with dilated convolutions and attention mechanism
def build_denoising_autoencoder(input_shape):
    input_layer = layers.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv1D(128, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(input_layer)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(64, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(32, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x)
    
    
    # Attention mechanism
    attention = layers.Conv1D(32, kernel_size=1, padding='same', activation='sigmoid')(x)
    x = layers.multiply([x, attention])
    
    # Decoder
    x = layers.Conv1D(32, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, dilation_rate=2, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(size=2)(x)
    
    
    output_layer = layers.Conv1D(1, kernel_size=3, padding='same', activation='linear')(x)
    
    autoencoder = models.Model(input_layer, output_layer)
    return autoencoder

input_shape = (512, 1)
autoencoder = build_denoising_autoencoder(input_shape)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.summary()

# Reshape data
x_train_noisy = np.expand_dims(x_train_noisy, axis=2)
x_train_clean = np.expand_dims(x_train_clean, axis=2)
x_val_noisy = np.expand_dims(x_val_noisy, axis=2)
x_val_clean = np.expand_dims(x_val_clean, axis=2)
x_test_noisy = np.expand_dims(x_test_noisy, axis=2)
x_test_clean = np.expand_dims(x_test_clean, axis=2)

# Train the model
history = autoencoder.fit(
    x_train_noisy, x_train_clean,
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_data=(x_val_noisy, x_val_clean)
)

# Save the model
model_save_path = 'denoising_autoencoder_model.h5'
autoencoder.save(model_save_path)
print(f'Model saved to {model_save_path}')

# Save training and validation loss
history_dict = history.history
loss_save_path = 'training_history.json'
with open(loss_save_path, 'w') as f:
    json.dump(history_dict, f)
print(f'Training history saved to {loss_save_path}')

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss_plot.png')
plt.show()

# Evaluate the model
denoised_data = autoencoder.predict(x_test_noisy)
mse = np.mean(np.square(x_test_clean - denoised_data))
rrmse = np.sqrt(mse) / np.sqrt(np.mean(np.square(x_test_clean)))
correlation_coefficient = np.mean([pearsonr(x_test_clean[i].flatten(), denoised_data[i].flatten())[0] for i in range(x_test_clean.shape[0])])

print(f'Mean MSE: {mse}')
print(f'Relative Root Mean Square Error (RRMSE): {rrmse}')
print(f'Mean Correlation Coefficient: {correlation_coefficient}')

# Plot denoised and original test signals
plt.figure(figsize=(15, 5))
plt.plot(x_test_clean[0], label='Original Signal')
plt.plot(denoised_data[0], label='Denoised Signal')
plt.title('Original vs. Denoised Test Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
