#!/usr/bin/env python
# coding: utf-8

# ## D Vamsidhar - 24070149005
# ### Autoencoder Assignment - Time Series Sensor Data

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load and preprocess data
df = pd.read_csv('sensor.csv')
df.drop(columns=['sensor_15', 'Unnamed: 0', 'machine_status'], inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df.drop(columns=['timestamp'], inplace=True)
df.fillna(method='ffill', inplace=True)

# Normalize the sensor data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# Define Autoencoder model
input_dim = data_scaled.shape[1]
encoding_dim = 16
input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation="relu")(input_layer)
encoder = Dense(32, activation="relu")(encoder)
bottleneck = Dense(encoding_dim, activation="relu")(encoder)
decoder = Dense(32, activation="relu")(bottleneck)
decoder = Dense(64, activation="relu")(decoder)
output_layer = Dense(input_dim, activation="linear")(decoder)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
history = autoencoder.fit(data_scaled, data_scaled, epochs=50, batch_size=64, validation_split=0.2, verbose=1)

# Anomaly detection
reconstructed_data = autoencoder.predict(data_scaled)
reconstruction_error = np.mean(np.abs(data_scaled - reconstructed_data), axis=1)
threshold = np.percentile(reconstruction_error, 95)
anomalies = reconstruction_error > threshold

# Plot reconstruction error
plt.figure(figsize=(10, 5))
plt.hist(reconstruction_error, bins=50, alpha=0.7, color='blue', label='Reconstruction Error')
plt.axvline(threshold, color='red', linestyle='--', label='Anomaly Threshold')
plt.legend()
plt.title('Reconstruction Error Distribution')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.show()

print(f"Detected {np.sum(anomalies)} anomalies out of {len(anomalies)} total data points.")


# In[7]:


# Visualize loss
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_training_history(history)


# ### CONCLUSION:
# 
# <ul>
#     <li>The implemented autoencoder successfully identifies anomalies by learning patterns in sensor data.
#     <li>Detected deviations based on reconstruction errors, enabling timely identification of potential issues in industrial pipelines.
#     <li>The training loss and validation loss graphs help assess the model's learning efficiency.
#     <li>The framework can be scaled and adapted to different industrial settings by adjusting model complexity, preprocessing techniques, and anomaly thresholds.
# </ul>
