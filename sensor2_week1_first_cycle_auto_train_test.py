"""
This script performs regression analysis on sensor data collected from a folder of CSV files.
It imports necessary libraries, reads the data, organizes it by temperature, and performs regression analysis using PCA.
The results are then plotted and saved in the specified directory.
"""
from sklearn.preprocessing import MinMaxScaler

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



# Extract wavelengths from the sample file
data = pd.read_csv(os.path.join(folder, file), delimiter=",", skiprows=[0], header=None, names=["Wavelength", "Intensity"])
wavelengths = data['Wavelength'].values

# Now you have both the intensity values and the wavelengths
print(train_data.shape)
print(test_data.shape)
print(wavelengths)



# Load the dictionary from the pickle file
with open('groups_of_files_week1_combined.pkl', 'rb') as f:
    groups_of_files_by_temp_and_id_week1 = pickle.load(f)
    
import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd

# Define the process_group_of_files function
def process_group_of_files(group_files, folder):
    delimiter = ","
    selected_variable_names = ["Wavelength", "Intensity"]
    intensities = []
    wavelengths = None

    for file in group_files:
        data = pd.read_csv(os.path.join(folder, file), delimiter=delimiter, skiprows=[0], header=None, names=selected_variable_names)
        if wavelengths is None:
            wavelengths = data['Wavelength'].values
        intensities.append(data['Intensity'].values)

    return intensities, wavelengths

# Prepare the data
def prepare_data(files_dict, folder):
    data = []
    for files in files_dict.values():
        for file in files:
            intensities, wavelengths = process_group_of_files([file], folder)
            data.extend(intensities)
    return np.array(data)

# Load the dictionary from the pickle file
with open('groups_of_files_week1_combined.pkl', 'rb') as f:
    groups_of_files_by_temp_and_id_week1 = pickle.load(f)

# Initialize dictionaries for training and testing sets
train_files_by_temp_and_id_week1 = {}
test_files_by_temp_and_id_week1 = {}

# Initialize a counter for the total number of files
total_files = 0

# Loop over the temperature and identifier tuples
for temp_id_tuple, groups_of_files_week1_combined in groups_of_files_by_temp_and_id_week1.items():
    train_files = []
    test_files = []

    for group_id, group_files in groups_of_files_week1_combined.items():
        total_files += len(group_files)
        split_index = int(len(group_files) * 0.8)
        train_files.extend(group_files[:split_index])
        test_files.extend(group_files[split_index:])

    train_files_by_temp_and_id_week1[temp_id_tuple] = train_files
    test_files_by_temp_and_id_week1[temp_id_tuple] = test_files

total_train_files = sum(len(files) for files in train_files_by_temp_and_id_week1.values())
total_test_files = sum(len(files) for files in test_files_by_temp_and_id_week1.values())

print(f"Total number of files: {total_files}")
print(f"Total number of training files: {total_train_files}")
print(f"Total number of testing files: {total_test_files}")

# Folder where the files are stored

# Prepare the training and testing data
train_data,wavelengths = prepare_data(train_files_by_temp_and_id_week1, folder)
test_data,wavelengths = prepare_data(test_files_by_temp_and_id_week1, folder)

# Normalize the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define the autoencoder model
input_dim = train_data.shape[1]
encoding_dim = 5

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)
decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(train_data, train_data, epochs=50, batch_size=256, shuffle=True, validation_data=(test_data, test_data))

# Evaluate the model
loss = autoencoder.evaluate(test_data, test_data)
print(f"Test loss: {loss}")
train_reconstructed = autoencoder.predict(train_data)
train_error = mean_squared_error(train_data, train_reconstructed)
print(f"Training error (MSE): {train_error}") # 0.009793665100736251

test_reconstructed = autoencoder.predict(test_data)
test_error = mean_squared_error(test_data, test_reconstructed)
print(f"Testing error (MSE): {test_error}") # 0.009791323549253848

# Use the autoencoder to encode and decode some data
encoded_data = autoencoder.predict(test_data)

specific_key = (-10.0, 'drywell_temp_')  # 

# Check if the specific key exists in the dictionary
if specific_key in groups_of_files_by_temp_and_id_week1:
    # Get the first group of files for the specific key
    first_group_files = groups_of_files_by_temp_and_id_week1[specific_key][1]  # Group 1


    # Process the first group of files
    intensities, wavelengths = process_group_of_files(first_group_files, folder)

    # Print the intensities and wavelengths
    print("Intensities:", intensities)
    print("Wavelengths:", wavelengths)
else:
    print(f"Key {specific_key} not found in the dictionary.")
    
import pickle
from sklearn.model_selection import train_test_split

# Load the data
with open('groups_of_files_week1_combined.pkl', 'rb') as f:
    groups_of_files_by_temp_and_id_week1 = pickle.load(f)


# Initialize dictionaries for training and testing sets
train_files_by_temp_and_id_week1 = {}
test_files_by_temp_and_id_week1 = {}

# Initialize a counter for the total number of files
total_files = 0

# Loop over the temperature and identifier tuples
for temp_id_tuple, groups_of_files_week1_combined in groups_of_files_by_temp_and_id_week1.items():
    # Initialize lists to store training and testing files
    train_files = []
    test_files = []

    # Split each group into training and testing sets
    for group_id, group_files in groups_of_files_week1_combined.items():
        total_files += len(group_files)  # Count the total number of files
        split_index = int(len(group_files) * 0.8)  # 80% for training, 20% for testing
        train_files.extend(group_files[:split_index])
        test_files.extend(group_files[split_index:])

    # Store the training and testing sets in their respective dictionaries
    train_files_by_temp_and_id_week1[temp_id_tuple] = train_files
    test_files_by_temp_and_id_week1[temp_id_tuple] = test_files

# Calculate the total number of training and testing files
total_train_files = sum(len(files) for files in train_files_by_temp_and_id_week1.values())
total_test_files = sum(len(files) for files in test_files_by_temp_and_id_week1.values())

# Print the total number of files and the number of training and testing files
print(f"Total number of files: {total_files}")
print(f"Total number of training files: {total_train_files}")
print(f"Total number of testing files: {total_test_files}")


import pickle
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import os
import pandas as pd

# Define the process_group_of_files function
def process_group_of_files(group_files, folder):
    delimiter = ","
    selected_variable_names = ["Wavelength", "Intensity"]
    intensities = []
    wavelengths = None

    for file in group_files:
        data = pd.read_csv(os.path.join(folder, file), delimiter=delimiter, skiprows=[0], header=None, names=selected_variable_names)
        if wavelengths is None:
            wavelengths = data['Wavelength'].values
        intensities.append(data['Intensity'].values)

    return intensities, wavelengths

# Prepare the data
def prepare_data(files_dict, folder):
    data = []
    for files in files_dict.values():
        for file in files:
            intensities, wavelengths = process_group_of_files([file], folder)
            data.extend(intensities)
    return np.array(data)

# Load the dictionary from the pickle file
with open('groups_of_files_week1_combined.pkl', 'rb') as f:
    groups_of_files_by_temp_and_id_week1 = pickle.load(f)

# Initialize dictionaries for training and testing sets
train_files_by_temp_and_id_week1 = {}
test_files_by_temp_and_id_week1 = {}

# Initialize a counter for the total number of files
total_files = 0

# Loop over the temperature and identifier tuples
for temp_id_tuple, groups_of_files_week1_combined in groups_of_files_by_temp_and_id_week1.items():
    train_files = []
    test_files = []

    for group_id, group_files in groups_of_files_week1_combined.items():
        total_files += len(group_files)
        split_index = int(len(group_files) * 0.8)
        train_files.extend(group_files[:split_index])
        test_files.extend(group_files[split_index:])

    train_files_by_temp_and_id_week1[temp_id_tuple] = train_files
    test_files_by_temp_and_id_week1[temp_id_tuple] = test_files

total_train_files = sum(len(files) for files in train_files_by_temp_and_id_week1.values())
total_test_files = sum(len(files) for files in test_files_by_temp_and_id_week1.values())

print(f"Total number of files: {total_files}")
print(f"Total number of training files: {total_train_files}")
print(f"Total number of testing files: {total_test_files}")

# Folder where the files are stored
folder = 'path_to_your_data_folder'

# Prepare the training and testing data
train_data = prepare_data(train_files_by_temp_and_id_week1, folder)
test_data = prepare_data(test_files_by_temp_and_id_week1, folder)

# Extract temperatures for regression targets
train_temperatures = np.array([temp_id_tuple[0] for temp_id_tuple in train_files_by_temp_and_id_week1.keys() for _ in train_files_by_temp_and_id_week1[temp_id_tuple]])
test_temperatures = np.array([temp_id_tuple[0] for temp_id_tuple in test_files_by_temp_and_id_week1.keys() for _ in test_files_by_temp_and_id_week1[temp_id_tuple]])

# Normalize the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define the autoencoder model
input_dim = train_data.shape[1]
encoding_dim = 5

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)
decoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the model
autoencoder.fit(train_data, train_data, epochs=50, batch_size=256, shuffle=True, validation_data=(test_data, test_data))

# Evaluate the model
loss = autoencoder.evaluate(test_data, test_data)
print(f"Test loss: {loss}")

# Encode the data
encoder = Model(input_layer, encoded)
encoded_train_data = encoder.predict(train_data)
encoded_test_data = encoder.predict(test_data)

# Train a regression model
regressor = LinearRegression()
regressor.fit(encoded_train_data, train_temperatures)

# Predict temperatures
train_predictions = regressor.predict(encoded_train_data)
test_predictions = regressor.predict(encoded_test_data)

# Calculate training and testing errors
train_mse = mean_squared_error(train_temperatures, train_predictions)
test_mse = mean_squared_error(test_temperatures, test_predictions)

print(f"Training error (MSE): {train_mse}")
print(f"Testing error (MSE): {test_mse}")

# projecting centered data onto the 10 principal components
# Assume that allTemps is a 1D array with the same length as file_comps
allTemps_week1 = allTemps_week1.reshape(-1, 1)

train_data= train_files
test_data=test_files
# Normalize the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data) # normalized to range of [0, 1]

# Define the autoencoder model
input_dim = train_data.shape[1]  # Input layer has the neurons equal to number of features in train_data  (10)
encoding_dim = 5 #  input data will be compressed into a 5-dimensional space. (Latent space)

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dropout(0.2)(encoded)  # Add dropout for regularization # encoding part of the autoencoder,
decoded = Dense(encoding_dim, activation='relu')(encoded)# takes (output of the encoding part) as input and starts the process of reconstructing the original input data.
decoded = Dense(input_dim, activation='sigmoid')(decoded)# takes the output of the previous decoding layer as input and produces the final reconstructed output.

# Autoencoder:  includes both the encoding and decoding parts. 
# It takes the input data, compresses it, and then reconstructs it.
#Encoder: includes only encoding part. It takes the input data and compresses it into a lower-dimensional representation.
#Autoencoder: For training the entire autoencoder to learn both encoding and decoding.
#Encoder: For extracting the encoded (compressed) representation of the input data after the autoencoder has been trained.

autoencoder = Model(input_layer, decoded) # creates the autoencoder model. #  input_layer as its input and encoded as its output.

encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the autoencoder on the training data
history = autoencoder.fit(train_data, train_data, epochs=100, batch_size=256, shuffle=True, validation_split=0.2)

# Get the encoded representations (loadings) for the training data
loadings_train = encoder.predict(train_data) #  uses the encoder part of the autoencoder to generate the compressed representations of the training data after the autoencoder has been trained.

# Create and fit the regression model using the loadings from the training data and the corresponding temperatures
reg = LinearRegression()
reg.fit(loadings_train, train_temps)

# Get the encoded representations (loadings) for the test data
loadings_test = encoder.predict(test_data)

# Make predictions on the test data
predictions_test = reg.predict(loadings_test)

# Compute the RMSE for the out-of-sample predictions
rmse_out_sample = np.sqrt(mean_squared_error(test_temps, predictions_test))
print(f"Out-of-sample RMSE: {rmse_out_sample}")
#encod dim =10 with data projected onto 10 principal components:  1.6361953008016663
# encod dim =5 with data projected onto 10 principal components: 4.85957184668906

predictions_train = reg.predict(loadings_train)

# Compute the RMSE for the in-sample predictions
rmse_in_sample = np.sqrt(mean_squared_error(train_temps, predictions_train))
print(f"In-sample RMSE: {rmse_in_sample}")
# encod dim =5 with data projected onto 10 principal components:  4.858672883444524
#encod dim =10 with data projected onto 10 principal components:   1.6371792639486755

##The training and test data are normalized using MinMaxScaler to scale the features to the range [0, 1].


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


# Compute the reconstruction error (MSE)
reconstruction_error = np.mean(np.square(averages_normalized - reconstructed_data)) # 0.6312682882351813with 10 loadings
# 0.6546029466366637 with 5 loadings