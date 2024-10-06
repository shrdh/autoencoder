from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
    
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, UpSampling1D, Reshape, Conv1DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import register_keras_serializable

# Load the training data from the file
with open('train_data.pkl', 'rb') as train_file:
    train_data = pickle.load(train_file)

# Load the testing data from the file
with open('test_data.pkl', 'rb') as test_file:
    test_data = pickle.load(test_file)


# Load train temperatures from the file
with open('train_temperatures.pkl', 'rb') as train_file:
    train_temperatures = pickle.load(train_file)

# Load test temperatures from the file
with open('test_temperatures.pkl', 'rb') as test_file:
    test_temperatures = pickle.load(test_file)
    

@register_keras_serializable()
# from tensorflow.keras import layers, Model
class ConvAutoencoder(Model):
    def __init__(self, input_shape, latent_dim):
        super(ConvAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Flatten()(x)
        encoded = layers.Dense(self.latent_dim)(x)
        return Model(inputs, encoded, name="encoder")

    def build_decoder(self):
        encoded_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense((self.input_shape[0] // 8) * 128)(encoded_inputs)
        x = layers.Reshape((self.input_shape[0] // 8, 128))(x)
        
        x = layers.Conv1DTranspose(128, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        
        x = layers.Conv1DTranspose(64, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        
        x = layers.Conv1DTranspose(32, 3, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
        
        x = layers.Conv1DTranspose(1, 3, activation='sigmoid', padding='same')(x)
        
        if x.shape[1] < self.input_shape[0]:
            x = layers.ZeroPadding1D(padding=(2, 2))(x)  # Adjust the dimensions to match the input shape
        elif x.shape[1] > self.input_shape[0]:
            x = layers.Cropping1D(cropping=(2, 2))(x)
        return Model(encoded_inputs, x, name="decoder")

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded# Manually apply min-max scaling
train_min = np.min(train_data)
train_max = np.max(train_data)

train_scaled = (train_data - train_min) / (train_max - train_min)
test_scaled = (test_data - train_min) / (train_max - train_min)

train_scaled = train_scaled.reshape((train_scaled.shape[0], train_scaled.shape[1], 1))  # (num_samples, height, width, channels)
test_scaled = test_scaled.reshape((test_scaled.shape[0], test_scaled.shape[1], 1))


# Define the input shape as a tuple of integers
input_shape = (train_scaled.shape[1], 1)  # Correct input shape
latent_dim = 50  # Adjust the latent dimension
autoencoder = ConvAutoencoder(input_shape, latent_dim)
autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

# # Implement early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = autoencoder.fit(train_scaled, train_scaled, epochs=30, batch_size=256, validation_split=0.2)


# Save the model
autoencoder.save('conv_autoencoder_model.h5')
# Calculate reconstruction error for each sample in the test dataset
reconstructed_data = autoencoder.predict(test_scaled)  # internally uses both encoder and decoder

# Calculate the overall reconstruction error (Mean Squared Error)
overall_reconstruction_error = np.mean(np.square(test_scaled - reconstructed_data))

# Print the overall reconstruction error
print("Overall reconstruction error (MSE) for the test dataset:")
print(overall_reconstruction_error)
    
# Man
# # Implement early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Train the model
# history = autoencoder.fit(X_train, X_train, epochs=30, batch_size=256, validation_split=0.2)

history = autoencoder.fit(train_scaled, train_scaled, epochs=30,batch_size=256,validation_split=0.2)



#Calculate reconstruction error for each sample in the test dataset
reconstructed_data = autoencoder.predict(test_scaled) # internally uses both encoder and decoder
overall_reconstruction_error = np.mean(np.square(test_scaled - reconstructed_data))

# Print the overall reconstruction error
print("Overall reconstruction error (MSE) for the test dataset:")
print(overall_reconstruction_error) 



# Save the entire model to a HDF5 file
autoencoder.save('laten_dim_1000.h5')

# Load the model from the HDF5 file
loaded_model = load_model('laten_dim_1000.h5')


# #  input data and its reconstruction after the autoencoder has been trained and processed the input data.
encoded_data = autoencoder.encoder(test_scaled).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy() # passing encoded_data through the decoder part of the autoencoder.

# #  input data and its reconstruction after the autoencoder has been trained and processed the input data.
plt.plot(np.arange(test_scaled[0].shape[0]),test_scaled[0],'*')
plt.plot(np.arange(test_scaled[0].shape[0]),decoded_data[0],'o')
plt.legend(labels=["Input", "Reconstruction"])
plt.show()


# # Encode the data
encoded_train_data = autoencoder.encoder.predict(train_scaled)
encoded_test_data = autoencoder.encoder.predict(test_scaled)

# Train a regression model
regressor = LinearRegression()

regressor.fit(encoded_train_data, train_temperatures)

# Predict temperatures
train_predictions = regressor.predict(encoded_train_data)
test_predictions = regressor.predict(encoded_test_data)

# Calculate training and testing errors
train_mse = np.sqrt(mean_squared_error(train_temperatures, train_predictions))
test_mse = np.sqrt(mean_squared_error(test_temperatures, test_predictions))

print(f"Training error (MSE): {train_mse}") 
print(f"Testing error (MSE): {test_mse}") #


# Assuming train_scaled and test_scaled are already defined and scaled
# Assuming train_temperatures and test_temperatures are already defined

# Encode the data


# Randomly select 2000 samples from the training data
num_samples = 10000
random_indices = np.random.choice(train_scaled.shape[0], num_samples, replace=False)
sampled_train_data = train_scaled[random_indices]
sampled_train_temperatures = train_temperatures[random_indices]

encoded_train_data = autoencoder.encoder.predict(sampled_train_data)
encoded_test_data = autoencoder.encoder.predict(test_scaled)

# Train a regression model
regressor = LinearRegression()
regressor.fit(encoded_train_data, sampled_train_temperatures)

# Predict temperatures
train_predictions = regressor.predict(encoded_train_data)
test_predictions = regressor.predict(encoded_test_data)

# Calculate training and testing errors
train_mse = np.sqrt(mean_squared_error(sampled_train_temperatures, train_predictions))
test_mse = np.sqrt(mean_squared_error(test_temperatures, test_predictions))

print(f"Training error (MSE): {train_mse}") 
print(f"Testing error (MSE): {test_mse}")

# Training error (MSE): 1.0424697714328202
# Testing error (MSE): 1.086318563883754 # 2000 samples # latent dim 1000 30 epochs

# Training error (MSE): 1.0485786245361561
#Testing error (MSE): 1.0676957583759026 # 4000 samples

# Training error (MSE): 1.0270155084735202
#Testing error (MSE): 1.0649177342378036 # 5000 samples

