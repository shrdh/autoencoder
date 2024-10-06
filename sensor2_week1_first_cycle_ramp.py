"""
This script performs regression analysis on sensor data collected from a folder of CSV files.
It imports necessary libraries, reads the data, organizes it by temperature, and performs regression analysis using PCA.
The results are then plotted and saved in the specified directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from getdata import get_data
from Regression import Regression
from plotter import *
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from datetime import datetime


# # Define the folder path
# folder_week3= r'D:\Downloads\sensor2_week3_ageing_and_aftermath\sensor2_week3_ageing_and_aftermath'

# folder=r'D:\Downloads\sensor2_12192023week1 (1)\sensor2_12192023week1\first_cycle'

# files = os.listdir(folder)



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
     
# files_by_temperature_week1_combined = {}  # Dictionary to store the files by temperature and format

# for file in files:
#     format_key = ""  # Initialize format key
#     # Check for both "drywell_temp_" and "temp_" and set the start index accordingly
#     if "drywell_temp_" in file:
#         start = file.find("drywell_temp_") + len("drywell_temp_")
#         format_key = "drywell_temp_"  # Set format key for drywell_temp_
#     elif "temp_" in file:
#         start = file.find("temp_") + len("temp_")
#         format_key = "temp_"  # Set format key for temp_
#     else:
#         continue  # Skip the file if it doesn't contain either prefix

#     end = file.find("_", start)
#     temperature = file[start:end]
#     temperature = temperature.replace(",", ".")  # Replace comma with dot for consistency
#     temperature_float = float(temperature)

#     # Create a combined key of temperature and format
#     combined_key = (temperature_float, format_key)

#     # Add the file to the dictionary under the corresponding combined key
#     if combined_key not in files_by_temperature_week1_combined:
#         files_by_temperature_week1_combined[combined_key] = []
    # files_by_temperature_week1_combined[combined_key].append(file)
 
 # Save the modified dictionary to a pickle file
# with open('sensor2_week1_combined.pickle', 'wb') as handle:
#     pickle.dump(files_by_temperature_week1_combined, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Open the pickle file and load its contents into a variable
with open('sensor2_week1_combined.pickle', 'rb') as handle:
    files_by_temperature_week1_combined = pickle.load(handle)
    


# Extract temperatures and sort them to maintain chronological order
sorted_temperatures = sorted(files_by_temperature_week1_combined.keys())

# Function to divide temperatures into ramp up and ramp down
def divide_ramp(temperatures):
    peak_index = np.argmax(temperatures)
    ramp_up = temperatures[:peak_index+1]
    ramp_down = temperatures[peak_index:]
    return ramp_up, ramp_down

# Divide the sorted temperatures into ramp up and ramp down
ramp_up_temps, ramp_down_temps = divide_ramp(sorted_temperatures)

# Initialize lists to store ramp up and ramp down files
ramp_up_files = []
ramp_down_files = []

# Collect files for ramp up temperatures
for temperature in ramp_up_temps:
    ramp_up_files.extend(files_by_temperature_week1_combined[temperature])

# Collect files for ramp down temperatures
for temperature in ramp_down_temps:
    ramp_down_files.extend(files_by_temperature_week1_combined[temperature])

# Print the number of files in ramp up and ramp down
print(f"Number of files in ramp up: {len(ramp_up_files)}")
print(f"Number of files in ramp down: {len(ramp_down_files)}")

# Initialize dictionaries to store ramp up and ramp down files
ramp_up_files_dict = {}
ramp_down_files_dict = {}

# Collect files for ramp up temperatures
for temperature in ramp_up_temps:
    ramp_up_files_dict[temperature] = files_by_temperature_week1_combined[temperature]

# Collect files for ramp down temperatures
for temperature in ramp_down_temps:
    ramp_down_files_dict[temperature] = files_by_temperature_week1_combined[temperature]

# Example usage: Accessing files for a specific temperature
# ramp_up_files_dict[temperature]
# ramp_down_files_dict[temperature]

# Function to process the intensities and wavelengths for ramp-up files
def process_ramp_up_files(ramp_up_files_dict, folder):
    # Define import options
    delimiter = ","
    selected_variable_names = ["Wavelength", "Intensity"]

    # Initialize lists to store the variables for the files
    intensities = []
    wavelengths = None

    # Read each file and calculate the intensities
    for file in ramp_up_files:
        data = pd.read_csv(os.path.join(folder, file), delimiter=delimiter, skiprows=[0], header=None, names=selected_variable_names)
        if wavelengths is None:
            wavelengths = data['Wavelength'].values
        intensities.append(data['Intensity'].values)

    return intensities, wavelengths

# ramp_down_intensities, ramp_down_wavelengths = process_files_for_temperature(ramp_down_files, folder)
print(np.shape(files))

import pickle



# Organizing a large number of files by temperature and identifier into manageable groups of up to 100 files each
# Organizing ramp-up files by temperature and identifier into manageable groups of up to 100 files each
groups_of_files_by_ramp = {}

# Loop over the temperature and identifier tuples
for temp_id_tuple in ramp_up_files_dict.keys():
    # Get the ramp-up files for the current temperature and identifier
    ramp_up_files = ramp_up_files_dict[temp_id_tuple]

    # Initialize a dictionary to store the groups of ramp-up files
    groups_of_files_ramp_combined = {}

    # Loop over the ramp-up files in groups of 100
    for i in range(0, len(ramp_up_files), 100):
        # Get the current group of 100 ramp-up files
        group_files = ramp_up_files[i:i+100]

        # Add the current group of ramp-up files to the dictionary
        groups_of_files_ramp_combined[i//100 + 1] = group_files

    # Add the groups of ramp-up files for the current temperature and identifier to the outer dictionary
    groups_of_files_by_ramp[temp_id_tuple] = groups_of_files_ramp_combined
    
 
with open('groups_of_files_ramp_combined.pkl', 'wb') as f:
    pickle.dump(groups_of_files_by_ramp, f)    

# # Load the dictionary from the file
# with open('groups_of_files_week1.pkl', 'rb') as f:
#     groups_of_files_by_temperature_week1 = pickle.load(f)

# #Save the dictionary to a file using pickle  
with open('groups_of_files_ramp_combined.pkl', 'rb') as f:
    groups_of_files_by_ramp = pickle.load(f)

# # Load the dictionary from the file
# with open('groups_of_files_ramp_combined.pkl', 'rb') as f:
#    groups_of_files_by_ramp  = pickle.load(f)

    


    
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


# Initialize a dictionary to store the averages for week 1
averages_week1_ramp = {}

# Loop over the temperature and identifier tuples for ramp-up files
for temperature, groups_of_files in groups_of_files_by_ramp.items():
    # Get the first 100 groups of files
    No_of_groups = list(groups_of_files.items())[:100]
    
    for group_number, group_files in No_of_groups:
        # Process the files for the current group
        intensities, _ = process_group_of_files(group_files, folder)
        
        # Calculate the average for the current group
        avg_intensities = np.mean(intensities, axis=0)
        
        # Store the average
        if temperature not in averages_week1_ramp:
            averages_week1_ramp[temperature] = []
        averages_week1_ramp[temperature].append(avg_intensities)
# Save the dictionary to the specified file using pickle
with open('D:/Downloads/averages_week1_ramp.pkl', 'wb') as file:
    pickle.dump(averages_week1_ramp, file)

# # Convert the averages dictionary to a list of numpy arrays
averages_list = [avg for avg_values in averages_week1_ramp.values() for avg in avg_values]

# # # Compute the average of averages
avg_intensities = np.mean(averages_list, axis=0)
# Perform PCA on the averages
pca = PCA(n_components=20)  # or any number up to min(n_samples, n_features)
pca.fit(averages_list)

# # Get the first 5 principal components
components_week1_ramp= pca.components_[:5]

# save

with open('components_week1_ramp.pickle', 'wb') as handle:
    pickle.dump(components_week1_ramp, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Load the components from the pickle file

with open('components_week1_ramp.pickle', 'rb') as handle:
    components_week1_ramp = pickle.load(handle)

print(np.shape(files))

# Assuming components_week3 is loaded and ready to use
num_components = len(components_week1_ramp)
fig, axes = plt.subplots(num_components, 1, figsize=(10, 6*num_components))

for i, component in enumerate(components_week1_ramp):
    ax = axes[i]
    ax.plot(component)  # Removed the label parameter
    ax.set_xlabel('Intensity', fontsize=10)
    ax.set_ylabel('Wavelength', fontsize=10)
    ax.set_title(f'Mode {i+1} (Ramp Up (First Cycle))', fontsize=10)

plt.tight_layout()
plt.show()
    
# import matplotlib.pyplot as plt

# # Assuming components_week3 is loaded and ready to use
# num_components = len(components_week1_combined)
# fig, axes = plt.subplots(num_components, 1, figsize=(10, 6*num_components))

# for i, component in enumerate(components_week1_combined):
#     ax = axes[i]
#     ax.plot(component)  # Removed the label parameter
#     ax.set_xlabel('Intensity', fontsize=10)
#     ax.set_ylabel('Wavelength', fontsize=10)
#     ax.set_title(f'Mode {i+1} (First Cycle)', fontsize=10)

# plt.tight_layout()
# plt.show()


from sklearn.linear_model import LinearRegression


print(np.shape(files))
# results_first_cycle_combined = {}

# # Process each file
# for file_first_cycle in files:
#     # Load the file into a DataFrame
#     df = pd.read_csv(os.path.join(folder, file_first_cycle))

#     # Assume that 'Intensity' is the column containing the intensities
#     file_intensities = df['Intensity'].values.reshape(-1, 1)
#     # print(np.shape(file_intensities))
#     avg_intensities = avg_intensities.reshape(-1, 1)

#     #print(avg_intensities)
#     # print(np.shape(avg_intensities))
#     # print(np.shape(components_week3))
#     # Compute the result for the current file
#     file_comp_week1 = np.dot(components_week1_combined, (file_intensities - avg_intensities))

#     # Store the result
#     results_first_cycle_combined[file_first_cycle] = file_comp_week1
  
# with open('D:\\Downloads\\results_first_cycle_combined.pkl', 'wb') as f:
#     pickle.dump(results_first_cycle_combined, f)
with open('D:\\Downloads\\results_first_cycle_combined.pkl', 'rb') as f:
    results_first_cycle_combined = pickle.load(f)

# with open('D:\\Downloads\\results_week3.pkl', 'rb') as f:
#     results_week3 = pickle.load(f)
# Get a list of all file_comp arrays in results


# file_comps_week3 = list(results_week3.values())

# Combine all file_comp arrays in results into a single 2D array
file_comps = np.hstack(list(results_first_cycle_combined.values()))

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

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Calculate the R-squared value
r2 = r2_score(allTemps_week1, predictions)
print(f"R-squared: {r2}") # 0.9962025735489282
# Calculate the Mean Squared Error
mse = mean_squared_error(allTemps_week1, predictions)
print(f"Mean Squared Error: {mse}") #  0.3419196442121769
# Calculate the Root Mean Squared Error
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}") #  0.5847389539035149


# # Plot the first 5 principal components
for i, component in enumerate(components_week3):
    plt.plot(component, label=f'PC{i+1}')

plt.xlabel('Intensity')
plt.ylabel('Wavelength')
plt.title('First 5 Principal Components')
plt.legend()
plt.show()

# # Plot the first 20 singular values
# plt.plot(pca.singular_values_[:20], 'o-')
# plt.xlabel('Component')
# plt.ylabel('Singular Value')
# plt.title('First 20 Singular Values')
# #plt.savefig('First 20 Singular Values.png')
# plt.show()


# # # #Plotting clean spectrum for Specific No of Groups
# # #Loop over the temperatures
for temperature, groups_of_files in groups_of_files_by_temperature_week3.items():
    # Convert the dictionary items to a list an
    # 
    # d take the first two items
    No_of_groups = list(groups_of_files.items())[:100]

    # Loop over the first two groups of files
    for group_number, group_files in No_of_groups:
        # Process the files for the current group
        intensities, _ = process_group_of_files(group_files, folder)

        # Calculate the mean spectrum for the current group
        mean_spectrum = np.mean(intensities, axis=0)

        # Plot the mean spectrum for the current group
        #plt.plot(mean_spectrum, label=f'Temperature {temperature}, Group {group_number}')
        plt.plot(mean_spectrum, label=f'{temperature}')

# Add a legend
plt.legend()

# Show the plot
plt.show()
