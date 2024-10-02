from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

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




# Now train_data and train_temperatures contain half of the original data
# test_data_split and test_temperatures_split contain the other half
   

# Now train_data and train_temperatures contain half of the original data
# test_data_split and test_temperatures_split contain the other half
# Normalize the data
# train_data = train_data / np.max(train_data)
# test_data = test_data / np.max(test_data)


# Manually apply min-max scaling
train_min = np.min(train_data)
train_max = np.max(train_data)

train_scaled = (train_data - train_min) / (train_max - train_min)
test_scaled = (test_data - train_min) / (train_max - train_min)

# Define the Autoencoder class using Keras functional API
class Autoencoder(Model):
    def __init__(self, input_shape, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = self.build_encoder(input_shape, latent_dim)
        self.decoder = self.build_decoder(input_shape, latent_dim)

    def build_encoder(self, input_shape, latent_dim):
        inputs = layers.Input(shape=input_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        # x = layers.Dense(64, activation='relu')(x)
        # x = layers.Dense(64, activation='relu')(x)
        encoded = layers.Dense(latent_dim)(x)  
        return Model(inputs, encoded, name="encoder")
    
    def build_decoder(self, input_shape, latent_dim):
        encoded_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(64, activation='relu')(encoded_inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        # x = layers.Dense(256, activation='relu')(x)
        # x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(np.prod(input_shape))(x)  # number of units = product of the dimensions of the input shape to transform the data into a flat !D array
        decoded = layers.Reshape(input_shape)(x) # reshape output to  input shape; convert the flat, 1D array back into the multi-dimensional format that matches the original input data.
        return Model(encoded_inputs, decoded, name="decoder")
    
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate and compile the model
input_shape = (train_scaled.shape[1],)   # Replace with your input shape
latent_dim = 5 # Adjust the latent dimension
autoencoder = Autoencoder(input_shape, latent_dim)
# autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

# # Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# Train the model
history = autoencoder.fit(train_scaled, train_scaled, epochs=30,batch_size=256, validation_data=(test_scaled, test_scaled))

# Encode and decode the test data
encoded_data = autoencoder.encoder(test_scaled).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

#Calculate reconstruction error for each sample in the test dataset
reconstructed_data = autoencoder.predict(test_scaled) # internally uses both encoder and decoder
overall_reconstruction_error = np.mean(np.square(test_scaled - reconstructed_data))

# Print the overall reconstruction error
print("Overall reconstruction error (MSE) for the test dataset:")
print(overall_reconstruction_error) 

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


encoded_data = autoencoder.encoder(test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy() # passing encoded_data through the decoder part of the autoencoder.

#  input data and its reconstruction after the autoencoder has been trained and processed the input data.
plt.plot(np.arange(test_data[0].shape[0]),test_data[0],'*')
plt.plot(np.arange(test_data[0].shape[0]),decoded_data[0],'o')
plt.legend(labels=["Input", "Reconstruction"])
plt.show()




# Normalize the data using Min-Max scaling and LekyReLU activation function
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define the Autoencoder class using Keras functional API
class Autoencoder(Model):
    def __init__(self, input_shape, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = self.build_encoder(input_shape, latent_dim)
        self.decoder = self.build_decoder(input_shape, latent_dim)

    def build_encoder(self, input_shape, latent_dim):
        inputs = layers.Input(shape=input_shape)
        x = layers.Flatten()(inputs)
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.Dense(64)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        encoded = layers.Dense(latent_dim)(x)
        return Model(inputs, encoded, name="encoder")
    
    def build_decoder(self, input_shape, latent_dim):
        encoded_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(64)(encoded_inputs)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.Dense(128)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.Dense(256)(x)
        x = layers.LeakyReLU(negative_slope=0.2)(x)
        x = layers.Dense(np.prod(input_shape))(x)  # number of units = product of the dimensions of the input shape to transform the data into a flat 1D array
        decoded = layers.Reshape(input_shape)(x)  # reshape output to input shape; convert the flat, 1D array back into the multi-dimensional format that matches the original input data.
        return Model(encoded_inputs, decoded, name="decoder")

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate and compile the model
input_shape = (train_data.shape[1],)   # Replace with your input shape
latent_dim = 3  # Adjust the latent dimension
autoencoder = Autoencoder(input_shape, latent_dim)
autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = autoencoder.fit(train_data, train_data, epochs=30, batch_size=256, shuffle=True, validation_data=(test_data, test_data), callbacks=[early_stopping])

# Encode and decode the test data
encoded_data = autoencoder.encoder(test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

# Calculate reconstruction error for each sample in the test dataset
reconstructed_data = autoencoder.predict(test_data)  # internally uses both encoder and decoder
overall_reconstruction_error = np.mean(np.square(test_data - reconstructed_data))

# Print the overall reconstruction error
print("Overall reconstruction error (MSE) for the test dataset:")
print(overall_reconstruction_error)


#0.0013877136169832497 # encod = 10
# 0.0013848765966418917 # encod = 20
# 0.001385383778659078 # encod = 30
# Overall reconstruction error (MSE) for the test dataset:
#0.0013844709905233077 # encod = 40

# Max scaling and relu

#0.06837681307275512 # encod = 20
# 0.07847349985680917 # encod = 10
# 0.07491624627299263 # encod =30
#0.0861229668672281 # encod = 5
# 0.04734909629817474 # encod = 40

# Normalize and Relu
# Training error (MSE): 1.534583340575241
# Testing error (MSE): 1.5297783426963227 # encod=10
# Training error (MSE): 0.7853165235414408
# Testing error (MSE): 0.7808939592044273 # encod=15


# raining error (MSE): 4.208015453446432
# Testing error (MSE): 4.209728112437402 # encod=5

# Training error (MSE): 1.1545165398505661
#Testing error (MSE): 1.1360900489795784 # encod=20

#Training error (MSE): 41.60672062821738
#Testing error (MSE): 41.55257949242786 # encod=3

# raining error (MSE): 0.3478341949045096
#Testing error (MSE): 0.3447881688435007 # encod=30

#Training error (MSE): 0.592190951698725
#Testing error (MSE): 0.5924707719823168 # encod=40
# Training error (MSE): 0.2617948711041965 # encod=50
#Testing error (MSE): 0.2590837888936347
# Training error (MSE): 0.23394052025324094
#  Testig: 0.23307800939252787 # encod = 60


# Max scaling and relu
# Training error (MSE): 2.9890760404506924
# Testing error (MSE): 2.9865234832717924 # encod = 20

# Training error (MSE): 4.188946331724044
#Testing error (MSE): 4.181903045838487 # encod = 10

# Training error (MSE): 4.394075626470511
#Testing error (MSE): 4.38415865233634 # encod = 5

# Training error (MSE): 1.4959876873539522
# Testing error (MSE): 1.5027088662285302 # encod = 30

# Training error (MSE): 1.9909064193925257
#Testing error (MSE): 1.9876222313335516 # encod = 40



# Assuming you have your data loaded as 'data' and 'temperatures'




# Define the Autoencoder class using Keras functional API
class Autoencoder(Model):
    def __init__(self, input_shape, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = self.build_encoder(input_shape, latent_dim)
        self.decoder = self.build_decoder(input_shape, latent_dim)

    def build_encoder(self, input_shape, latent_dim):
        inputs = layers.Input(shape=input_shape)
        x = layers.Flatten()(inputs)
        x = self.dense_block(x, 256)
        x = self.dense_block(x, 128)
        x = self.dense_block(x, 64)
        encoded = layers.Dense(latent_dim)(x)
        return Model(inputs, encoded, name="encoder")
    
    def build_decoder(self, input_shape, latent_dim):
        encoded_inputs = layers.Input(shape=(latent_dim,))
        x = self.dense_block(encoded_inputs, 64)
        x = self.dense_block(x, 128)
        x = self.dense_block(x, 256)
        x = layers.Dense(np.prod(input_shape))(x)
        decoded = layers.Reshape(input_shape)(x)
        return Model(encoded_inputs, decoded, name="decoder")
    
    def dense_block(self, x, units):
        x = layers.Dense(units, kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        return layers.Dropout(0.3)(x)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# Instantiate and compile the model
input_shape =  (train_data.shape[1],) 
latent_dim = 1  # Adjust as needed
autoencoder = Autoencoder(input_shape, latent_dim)

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

autoencoder.compile(optimizer=optimizer, loss='mse')

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with K-fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_train_losses = []
fold_val_losses = []

for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
    print(f"Training on fold {fold+1}/{n_splits}")
    
    # Split data
    X_train_fold, X_val_fold = train_data[train_index], train_data[val_index]
    
    # Train the model
    history = autoencoder.fit(
        X_train_fold, X_train_fold,
        epochs=100,
        batch_size=256,
        validation_data=(X_val_fold, X_val_fold),
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Store the best losses
    fold_train_losses.append(min(history.history['loss']))
    fold_val_losses.append(min(history.history['val_loss']))

# Print average losses
print(f"Average training loss: {np.mean(fold_train_losses)}")
print(f"Average validation loss: {np.mean(fold_val_losses)}")

# Encode and decode the test data
encoded_data = autoencoder.encoder(test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

# Calculate reconstruction error for the test dataset
reconstructed_data = autoencoder.predict(test_data)
overall_reconstruction_error = np.mean(np.square(test_data - reconstructed_data))

print("Overall reconstruction error (MSE) for the test dataset:")
print(overall_reconstruction_error)

# Encode the data
encoded_train_data = autoencoder.encoder.predict(train_data)
encoded_test_data = autoencoder.encoder.predict(test_data)

# Train a regression model
regressor = LinearRegression()
regressor.fit(encoded_train_data, train_temperatures)

# Predict temperatures
train_predictions = regressor.predict(encoded_train_data)
test_predictions = regressor.predict(encoded_test_data)

# Calculate training and testing errors
train_mse = mean_squared_error(train_temperatures, train_predictions)
test_mse = mean_squared_error(test_temperatures, test_predictions)

print(f"Training MSE: {train_mse}")
print(f"Testing MSE: {test_mse}")


# Extract wavelengths from the sample file
data = pd.read_csv(os.path.join(folder, file), delimiter=",", skiprows=[0], header=None, names=["Wavelength", "Intensity"])
wavelengths = data['Wavelength'].values

# Now you have both the intensity values and the wavelengths
print(train_data.shape)
print(test_data.shape)
print(wavelengths)

import pickle

# Load the wavelengths from the file
with open('wavelengths.pkl', 'rb') as wavelength_file:
    wavelengths = pickle.load(wavelength_file)

print(wavelengths)

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the training data from the file
with open('train_data.pkl', 'rb') as train_file:
    train_data = pickle.load(train_file)

# Load the testing data from the file
with open('test_data.pkl', 'rb') as test_file:
    test_data = pickle.load(test_file)

# Load the wavelengths from the file
with open('wavelengths.pkl', 'rb') as wavelength_file:
    wavelengths = pickle.load(wavelength_file)

# Now you have both the intensity values and the wavelengths
print(train_data.shape)
print(test_data.shape)
print(wavelengths)

# ZPL wavelength (example value, replace with actual ZPL wavelength)
zpl_wavelength = 637  # Example: ZPL at 637 nm

# Find the pixel index closest to the ZPL wavelength
zpl_pixel_index = np.argmin(np.abs(wavelengths - zpl_wavelength))

# Find the pixel index for the flat 600 nm region
flat_600nm_index = np.argmin(np.abs(wavelengths - 600))

# Find the pixel index for the first phonon region (example: 700 nm)
phonon_region_index = np.argmin(np.abs(wavelengths - 700))

# Extract pixel data
train_zpl_pixel = train_data[:, zpl_pixel_index]
train_flat_600nm_pixel = train_data[:, flat_600nm_index]
train_phonon_region_pixel = train_data[:, phonon_region_index]

test_zpl_pixel = test_data[:, zpl_pixel_index]
test_flat_600nm_pixel = test_data[:, flat_600nm_index]
test_phonon_region_pixel = test_data[:, phonon_region_index]

# Apply min-max scaling
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

train_scaled_zpl_pixel = train_scaled[:, zpl_pixel_index]
train_scaled_flat_600nm_pixel = train_scaled[:, flat_600nm_index]
train_scaled_phonon_region_pixel = train_scaled[:, phonon_region_index]

test_scaled_zpl_pixel = test_scaled[:, zpl_pixel_index]
test_scaled_flat_600nm_pixel = test_scaled[:, flat_600nm_index]
test_scaled_phonon_region_pixel = test_scaled[:, phonon_region_index]

# Plot the data before and after scaling
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(train_zpl_pixel, label='Train ZPL Pixel')
plt.plot(train_flat_600nm_pixel, label='Train Flat 600nm Pixel')
plt.plot(train_phonon_region_pixel, label='Train Phonon Region Pixel')
plt.title('Train Data Before Scaling')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(train_scaled_zpl_pixel, label='Train ZPL Pixel (Scaled)')
plt.plot(train_scaled_flat_600nm_pixel, label='Train Flat 600nm Pixel (Scaled)')
plt.plot(train_scaled_phonon_region_pixel, label='Train Phonon Region Pixel (Scaled)')
plt.title('Train Data After Scaling')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(test_zpl_pixel, label='Test ZPL Pixel')
plt.plot(test_flat_600nm_pixel, label='Test Flat 600nm Pixel')
plt.plot(test_phonon_region_pixel, label='Test Phonon Region Pixel')
plt.title('Test Data Before Scaling')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(test_scaled_zpl_pixel, label='Test ZPL Pixel (Scaled)')
plt.plot(test_scaled_flat_600nm_pixel, label='Test Flat 600nm Pixel (Scaled)')
plt.plot(test_scaled_phonon_region_pixel, label='Test Phonon Region Pixel (Scaled)')
plt.title('Test Data After Scaling')
plt.legend()

plt.tight_layout()
plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the training data from the file
with open('train_data.pkl', 'rb') as train_file:
    train_data = pickle.load(train_file)

# Load the wavelengths from the file
with open('wavelengths.pkl', 'rb') as wavelength_file:
    wavelengths = pickle.load(wavelength_file)

# Find the average spectrum across all training samples
average_spectrum = np.mean(train_data, axis=0)

# Identify the ZPL wavelength by finding the wavelength with the maximum intensity
zpl_index = np.argmax(average_spectrum)
zpl_wavelength = wavelengths[zpl_index]

print(f"Identified ZPL Wavelength: {zpl_wavelength} nm")

# Plot the average spectrum and highlight the ZPL
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, average_spectrum, label='Average Spectrum')
plt.axvline(x=zpl_wavelength, color='r', linestyle='--', label=f'ZPL at {zpl_wavelength} nm')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.title('Average Spectrum with Identified ZPL')
plt.legend()
plt.show()

# Load the wavelengths from the file
with open('wavelengths.pkl', 'rb') as wavelength_file:
    wavelengths = pickle.load(wavelength_file)

# Find the average spectrum across all training samples
average_spectrum = np.mean(train_data, axis=0)

# Identify the ZPL wavelength by finding the wavelength with the maximum intensity
zpl_index = np.argmax(average_spectrum)
zpl_wavelength = wavelengths[zpl_index]
print(f"Identified ZPL Wavelength: {zpl_wavelength} nm")

# Find the pixel index closest to the ZPL wavelength
zpl_pixel_index = np.argmin(np.abs(wavelengths - zpl_wavelength))
print(f"ZPL Pixel Index: {zpl_pixel_index}, Wavelength: {wavelengths[zpl_pixel_index]}")

# Find the pixel index for the flat 600 nm region
flat_600nm_index = np.argmin(np.abs(wavelengths - 600))
print(f"Flat 600nm Pixel Index: {flat_600nm_index}, Wavelength: {wavelengths[flat_600nm_index]}")

# Find the indices for the first phonon region (wavelengths > 650 nm and < 750 nm)
phonon_region_indices = np.where((wavelengths > 650) & (wavelengths < 750))[0]
print(f"Phonon Region Indices: {phonon_region_indices}, Wavelengths: {wavelengths[phonon_region_indices]}")

# Extract pixel data
train_zpl_pixel = train_data[:, zpl_pixel_index]
train_flat_600nm_pixel = train_data[:, flat_600nm_index]
train_phonon_region_pixels = train_data[:, phonon_region_indices]

test_zpl_pixel = test_data[:, zpl_pixel_index]
test_flat_600nm_pixel = test_data[:, flat_600nm_index]
test_phonon_region_pixels = test_data[:, phonon_region_indices]

# Apply min-max scaling
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

train_scaled_zpl_pixel = train_scaled[:, zpl_pixel_index]
train_scaled_flat_600nm_pixel = train_scaled[:, flat_600nm_index]
train_scaled_phonon_region_pixels = train_scaled[:, phonon_region_indices]

test_scaled_zpl_pixel = test_scaled[:, zpl_pixel_index]
test_scaled_flat_600nm_pixel = test_scaled[:, flat_600nm_index]
test_scaled_phonon_region_pixels = test_scaled[:, phonon_region_indices]

# Plot the data before and after scaling
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(train_zpl_pixel, label='Train ZPL Pixel')
plt.plot(train_flat_600nm_pixel, label='Train Flat 600nm Pixel')
plt.plot(train_phonon_region_pixels.mean(axis=1), label='Train Phonon Region Pixels (Mean)')
plt.title('Train Data Before Scaling')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(train_scaled_zpl_pixel, label='Train ZPL Pixel (Scaled)')
plt.plot(train_scaled_flat_600nm_pixel, label='Train Flat 600nm Pixel (Scaled)')
plt.plot(train_scaled_phonon_region_pixels.mean(axis=1), label='Train Phonon Region Pixels (Scaled, Mean)')
plt.title('Train Data After Scaling')
plt.legend()

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(test_zpl_pixel, label='Test ZPL Pixel')
plt.plot(test_flat_600nm_pixel, label='Test Flat 600nm Pixel')
plt.plot(test_phonon_region_pixels.mean(axis=1), label='Test Phonon Region Pixels (Mean)')
plt.title('Test Data Before Scaling')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(test_scaled_zpl_pixel, label='Test ZPL Pixel (Scaled)')
plt.plot(test_scaled_flat_600nm_pixel, label='Test Flat 600nm Pixel (Scaled)')
plt.plot(test_scaled_phonon_region_pixels.mean(axis=1), label='Test Phonon Region Pixels (Scaled, Mean)')
plt.title('Test Data After Scaling')
plt.legend()

plt.tight_layout()
plt.show()