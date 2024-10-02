"""
This script performs regression analysis on sensor data collected from a folder of CSV files.
It imports necessary libraries, reads the data, organizes it by temperature, and performs regression analysis using PCA.
The results are then plotted and saved in the specified directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
import scipy.io as sio
from getdata import get_data
from plotter import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import pickle

from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.cm as cm
from datetime import datetime
from sklearn.decomposition import SparsePCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pickle
import os
import re

folder = r'D:\Downloads\sensor2_12192023week1 (1)\sensor2_12192023week1\first_cycle'
files = os.listdir(folder)

# # Regular expression to match laser power and frame number in the file name
pattern = re.compile(r'laser_power_(\d+)_.*-Frame-(\d+)')

for file in files:
    match = pattern.search(file)
    if match:
        power, frame = int(match.group(1)), int(match.group(2))
        # Path for the file to potentially delete
        file_path = os.path.join(folder, file)
        # Condition to delete files: 90% power or the first frame of 30% power measurements
        if power == 90 or (power == 30 and frame == 1):
            try:
                os.remove(file_path)
                print(f"Deleted: {file}")
            except OSError as e:
                print(f"Error deleting {file}: {e}")

print(np.shape(files))



# Assuming 'files' is a list of filenames

wavelengths = None
allTemps_week1 = []
allIntensities = []

# Define import options
delimiter = ","
selected_variable_names = ["Wavelength", "Intensity"]


# Load the allTemps dictionary
with open('D:\\Downloads\\allTemps_week1.pkl', 'rb') as f:
     allTemps_week1 = pickle.load(f)
     
# Open the pickle file and load its contents into a variable
with open('sensor2_week1_combined.pickle', 'rb') as handle:
    files_by_temperature_week1_combined = pickle.load(handle)
    

# # # Function to process the intensities and wavelengths for a specific temperature   
def process_files_for_temperature(files_by_temperature_week1_combined, temperature, folder):
    # Get the files for the specified temperature
    files = files_by_temperature_week1_combined.get(temperature, [])

    # Define import options
    delimiter = ","
    selected_variable_names = ["Wavelength", "Intensity"]

    # Initialize lists to store the variables for the files
    intensities = []
    wavelengths = None
   
    # Read each file and calculate the intensities
    for file in files:
        data = pd.read_csv(os.path.join(folder, file), delimiter=delimiter, skiprows=[0], header=None, names=selected_variable_names)
        if wavelengths is None:
            wavelengths = data['Wavelength'].values
        intensities.append(data['Intensity'].values)

    return intensities, wavelengths    
print(np.shape(files))


# Load the dictionary from the file
with open('groups_of_files_week1_combined.pkl', 'rb') as f:
   groups_of_files_by_temp_and_id_week1  = pickle.load(f)


# for temperature, groups in groups_of_files_by_temp_and_id_week1.items():
#     print(f"Temperature: {temperature}")
#     for group_number, files in groups.items():
#         print(f"  Group {group_number}: {len(files)} files")

    
# for every group of files processed, it returns a list of intensity arrays (one for each file) and a single wavelength array (assumed common across all files).
def process_group_of_files(group_files, folder):
   
    # Define import options
   

    delimiter = ","
    selected_variable_names = ["Wavelength", "Intensity"]

    # Initialize lists to store the variables for the files
    intensities = []
    wavelengths = None


    # Read each file and calculate the intensities
    for file in group_files:
        data = pd.read_csv(os.path.join(folder, file), delimiter=delimiter, skiprows=[0], header=None, names=selected_variable_names)
        if wavelengths is None:
            wavelengths = data['Wavelength'].values
        intensities.append(data['Intensity'].values)

    return intensities, wavelengths 
print(np.shape(files))




# Load the dictionary from the pickle file
with open('groups_of_files_week1_combined.pkl', 'rb') as f:
    groups_of_files_by_temp_and_id_week1 = pickle.load(f)

# specific_key = (-10.0, 'drywell_temp_')  # 

# # Check if the specific key exists in the dictionary
# if specific_key in groups_of_files_by_temp_and_id_week1:
#     # Get the first group of files for the specific key
#     first_group_files = groups_of_files_by_temp_and_id_week1[specific_key][1]  # Group 1


#     # Process the first group of files
#     intensities, wavelengths = process_group_of_files(first_group_files, folder)

#     # Print the intensities and wavelengths
#     print("Intensities:", intensities)
#     print("Wavelengths:", wavelengths)
# else:
#     print(f"Key {specific_key} not found in the dictionary.")

    
with open('D:/Downloads/averages_week1_combined.pkl', 'rb') as f:
    averages_week1_combined = pickle.load(f)   


# # Convert the averages dictionary to a list of numpy arrays


averages_list = [avg for avg_values in averages_week1_combined.values() for avg in avg_values]
# Autoencoder for Dimensionality Reduction


# Convert list to array
averages_array = np.array(averages_list)

# Normalize the data
scaler = StandardScaler()
averages_normalized = scaler.fit_transform(averages_array) # standardizes features by removing the mean and scaling to unit variance.

# Define the autoencoder model
input_dim = averages_normalized.shape[1] # Input layer has 1340 neurons
encoding_dim = 5 # 
input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)  # Add dropout for regularization
# encoded = Dense(encoding_dim // 2, activation='relu')(encoded)  # Latent SPace
decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)


autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')


# Train the autoencoder
history=autoencoder.fit(averages_normalized, averages_normalized, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)

# Get the encoded representations (loadings)
loadings = encoder.predict(averages_normalized)

reconstructed_data = autoencoder.predict(averages_normalized)

# Compute the reconstruction error (MSE)
reconstruction_error = np.mean(np.square(averages_normalized - reconstructed_data)) # 0.6312682882351813with 10 loadings
# 0.6546029466366637 with 5 loadings

# Compute the dot product
modes_10 = np.dot(np.transpose(loadings), averages_normalized)

# Plot the loadings
n_loadings_to_plot = 10 # Change this to plot more or fewer loadings
fig, axs = plt.subplots(n_loadings_to_plot, 1, figsize=(12, 4*n_loadings_to_plot), sharex=True)
fig.suptitle('First {} Autoencoder Loadings'.format(n_loadings_to_plot))

for i in range(n_loadings_to_plot):
    axs[i].plot(range(loadings.shape[0]), loadings[:, i])
    axs[i].set_ylabel('Loading {}'.format(i+1))

axs[-1].set_xlabel('Sample Index')
plt.tight_layout()
plt.show()


avg_intensities = np.mean(averages_list, axis=0)

with open('D:\\Downloads\\results_auto.pkl', 'rb') as f:
     results_auto = pickle.load(f)  # 10 modes
    
with open('D:\\Downloads\\results_auto_mode_5.pkl', 'rb') as f:
     results_auto_mode_5 = pickle.load(f)   # 5 modes

file_comps = np.hstack(list(results_auto_mode_5 .values())) # file comps  is projecting centered data onto the principal components

file_comps_ten = np.hstack(list(results_auto .values()))
# Assume that allTemps is a 1D array with the same length as file_comps
allTemps_week1 = allTemps_week1.reshape(-1, 1)


# Create a LinearRegression object
reg = LinearRegression()
# Fit the model to the data
reg.fit(file_comps.T, allTemps_week1)
# Make predictions on the training data
predictions = reg.predict(file_comps.T)

# Print the predictions and the actual target values
print(f"Predictions: {predictions}")
print(f"Actual: {allTemps_week1}")

# Calculate the R-squared value
r2 = r2_score(allTemps_week1, predictions)
print(f"R-squared: {r2}") #  0.9993994256841303(5 modes)
# Calculate the Mean Squared Error
mse = mean_squared_error(allTemps_week1, predictions)

print(f"Mean Squared Error: {mse}") #  0.2616837782516357 (with 10 modes); 0.49061584212618986(5modes)
# Calculate the Root Mean Squared Error;
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}") #   0.5115503672676188 (with 10 modes) # 0.700439749104939 (5modes)









# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 15))

for i, component in enumerate(modes_10):
    axes[i].plot(component, label=f'PC{i+1}')
    axes[i].set_xlabel('Intensity')
    axes[i].set_ylabel('Wavelength')
    axes[i].legend()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Plot the first 5 modes
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 15))
for i in range(5):
    axes[i].plot(modes_10[i], label=f'PC{i+1}')
    axes[i].set_xlabel('Intensity')
    axes[i].set_ylabel('Wavelength')
    axes[i].legend()
plt.tight_layout()
plt.savefig('first_5_modes.png')
plt.show()

# Plot the next 5 modes
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 15))
for i in range(5, 10):
    axes[i-5].plot(modes_10[i], label=f'PC{i+1}')
plt.tight_layout()
plt.savefig('next_5_modes.png')
plt.show()


# Compute the RMSE for the in-sample and out-sample predictions




# Transpose file_comps_clean so that each row represents a file
file_comps_ten = file_comps_ten.T

# Split the data into training and test sets
train_data = file_comps_ten[::2]  # Even-indexed files
train_temps = allTemps_week1[::2]  # Corresponding temperatures

test_data = file_comps_ten[1::2]  # Odd-indexed files
test_temps = allTemps_week1[1::2]  # Corresponding temperatures

# Define the autoencoder model
input_dim = train_data.shape[1]  # Input layer has the number of features in train_data
encoding_dim = 10  # Number of neurons in the encoding layer #

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)  # Add dropout for regularization
decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the autoencoder on the training data
history = autoencoder.fit(train_data, train_data, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)



# Get the encoded representations (loadings) for the training data
loadings_train = encoder.predict(train_data)

# Create and fit the regression model using the loadings from the training data and the corresponding temperatures
reg = LinearRegression()
reg.fit(loadings_train, train_temps)

# Get the encoded representations (loadings) for the test data
loadings_test = encoder.predict(test_data)

# Make predictions on the test data
predictions_test = reg.predict(loadings_test)

# Compute the RMSE for the out-of-sample predictions
rmse_out_sample = np.sqrt(mean_squared_error(test_temps, predictions_test))
print(f"Out-of-sample RMSE: {rmse_out_sample}")# 5 loadings with 5 modes :  3.889157677956472; 
# 20 loadings with 5 modes: 0.9057241590096896

predictions_train = reg.predict(loadings_train)

# Compute the RMSE for the in-sample predictions
rmse_in_sample = np.sqrt(mean_squared_error(train_temps, predictions_train))
print(f"In-sample RMSE: {rmse_in_sample}") # 5 loadings with  5 modes: 3.8942861110155937
    # 20 loadings with 5 modes : 0.9055012409173766




# Transpose file_comps_clean so that each row represents a file
file_comps = file_comps.T

# Split the data into training and test sets
train_data = file_comps[::2]  # Even-indexed files
train_temps = allTemps_week1[::2]  # Corresponding temperatures

test_data = file_comps[1::2]  # Odd-indexed files
test_temps = allTemps_week1[1::2]  # Corresponding temperatures

# Define the autoencoder model
input_dim = train_data.shape[1]  # Input layer has the number of features in train_data
encoding_dim = 20  # Number of neurons in the encoding layer #

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)  # Add dropout for regularization
decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the autoencoder on the training data
history = autoencoder.fit(train_data, train_data, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)

# autoencoder is trained using train data, after training the autoencoder, the encoder part of the autoencoder is used to get the encoded representations (loadings) for the train_data.

# Get the encoded representations (loadings) for the training data
loadings_train = encoder.predict(train_data)

# Create and fit the regression model using the loadings from the training data and the corresponding temperatures


# A linear regression model is created.
# The regression model is trained using the loadings from the training data (loadings_train) and the corresponding temperatures (train_temps).
# The encoder is used to get the encoded representations (loadings) for the test_data.

# The regression model is used to make predictions on the test data (loadings_test).
reg = LinearRegression()
reg.fit(loadings_train, train_temps)

# Get the encoded representations (loadings) for the test data
loadings_test = encoder.predict(test_data)

# Make predictions on the test data
predictions_test = reg.predict(loadings_test)

# Compute the RMSE for the out-of-sample predictions
rmse_out_sample = np.sqrt(mean_squared_error(test_temps, predictions_test))
print(f"Out-of-sample RMSE: {rmse_out_sample}")# 5 loadings with 5 modes :  3.889157677956472; 
# 20 loadings with 5 modes: 0.9057241590096896
# 10 loadings with 10 modes: 3.5528399235212658

predictions_train = reg.predict(loadings_train)

# Compute the RMSE for the in-sample predictions
rmse_in_sample = np.sqrt(mean_squared_error(train_temps, predictions_train))
print(f"In-sample RMSE: {rmse_in_sample}") # 5 loadings with  5 modes: 3.8942861110155937
    # 20 loadings with 5 modes : 0.9055012409173766
    # 10 loadings with 10 modes:  3.557482121544311



file_comps_ten = file_comps_ten.T

# Split the data into training and test sets
train_data = file_comps_ten[::2]  # Even-indexed files
train_temps = allTemps_week1[::2]  # Corresponding temperatures

test_data = file_comps_ten[1::2]  # Odd-indexed files
test_temps = allTemps_week1[1::2]  # Corresponding temperatures

# Scale the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Define the autoencoder model
input_dim = train_data_scaled.shape[1]  # Input layer has the number of features in train_data
encoding_dim = 10  # Number of neurons in the encoding layer

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)  # Add dropout for regularization
decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the autoencoder on the training data
history = autoencoder.fit(train_data_scaled, train_data_scaled, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)

# Get the encoded representations (loadings) for the training data
loadings_train = encoder.predict(train_data_scaled)

# Compute in-sample error (training data)
train_reconstructions = autoencoder.predict(train_data_scaled)
train_mse = mean_squared_error(train_data_scaled, train_reconstructions)
train_rmse = np.sqrt(train_mse)
print(f"In-Sample RMSE (Training): {train_rmse}") #  0.8115967757460912

# Compute out-of-sample error (test data)
test_reconstructions = autoencoder.predict(test_data_scaled)
test_mse = mean_squared_error(test_data_scaled, test_reconstructions)
test_rmse = np.sqrt(test_mse)
print(f"Out-of-Sample RMSE (Test): {test_rmse}") #0.8828397919569425

file_comps = np.hstack(list(results_auto_mode_5 .values())) 



file_comps = file_comps.T

# Split the data into training and test sets
train_data = file_comps[::2]  # Even-indexed files
train_temps = allTemps_week1[::2]  # Corresponding temperatures

test_data = file_comps[1::2]  # Odd-indexed files
test_temps = allTemps_week1[1::2]  # Corresponding temperatures

# Scale the data
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Define the autoencoder model
input_dim = train_data_scaled.shape[1]  # Input layer has the number of features in train_data
encoding_dim = 5  # Number of neurons in the encoding layer

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)  # Add dropout for regularization
decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the autoencoder on the training data
history = autoencoder.fit(train_data_scaled, train_data_scaled, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)

# Get the encoded representations (loadings) for the training data
loadings_train = encoder.predict(train_data_scaled)

# Compute in-sample error (training data)
train_reconstructions = autoencoder.predict(train_data_scaled)
train_mse = mean_squared_error(train_data_scaled, train_reconstructions)
train_rmse = np.sqrt(train_mse)
print(f"In-Sample RMSE (Training): {train_rmse}") #   0.8425841380898443

# Compute out-of-sample error (test data)
test_reconstructions = autoencoder.predict(test_data_scaled)
test_mse = mean_squared_error(test_data_scaled, test_reconstructions)
test_rmse = np.sqrt(test_mse)
print(f"Out-of-Sample RMSE (Test): {test_rmse}")  #0.8425728577753735

