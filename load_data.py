#%% Load the data to a np.array file to save time when running the code later
#Imports
import pydicom
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import numpy as np
import os
#%% read dicom files

base_path = "C:/Users/ariaa/Documents/Jobb/PhD søknad/UiO task/Python/manifest-1669817128730/Colorectal-Liver-Metastases"

studies_path = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

subject_data_list = []

i = 0
for studie in studies_path:
    sub_path = os.path.join(base_path, studie)
    sub_folder = os.listdir(sub_path)[0]
    sub_sub_folder = os.listdir(os.path.join(sub_path, sub_folder))
    for folder in sub_sub_folder: 
        if "NA" in folder:
            images_path = os.path.join(sub_path, sub_folder, folder)

    image_path = os.path.join(images_path, os.listdir(images_path)[0])

    ds = pydicom.dcmread(image_path)
    
    # Extract the image data
    image_data = ds.pixel_array

    # Get a list of all DICOM files in the folder
    dicom_files = [image for image in os.listdir(images_path) if image.endswith('.dcm')]

    # Initialize an empty list to store the image data
    image_data_list = []

    # Loop through each DICOM file
    for image in dicom_files:
        # Construct the file path
        image_path = os.path.join(images_path, image)
        
        # Load the DICOM file
        ds = pydicom.dcmread(image_path)
        
        # Extract the image data
        image_data = ds.pixel_array
        
        # Append the image data to the list
        image_data_list.append(image_data)

    # Convert the list of image data into a numpy array
    subject_data_list.append(image_data_list)

#%% Visualize as animation
'''
print("Image data array shape:", len(subject_data_list), len(subject_data_list[0]), len(subject_data_list[1]))

image_data_array = np.array(subject_data_list[1])

num_frames = image_data_array.shape[0]

# Initialize a figure for plotting
fig, ax = plt.subplots()

# Create an empty image plot
im = ax.imshow(image_data_array[0], cmap='gray')

def update(frame):
    # Update the image data for each frame
    im.set_data(image_data_array[frame])
    return im,

# Create an animation
anim = FuncAnimation(fig, update, frames=num_frames, interval=10, repeat=False, blit=True)

# Display the animation
plt.show()


# show the segmentation for the first subject


# Path to the DICOM file
# file_path = "C:/Users/ariaa/Documents/Jobb/PhD søknad/UiO task/Python/manifest-1669817128730/Colorectal-Liver-Metastases/CRLM-CT-1001/06-06-1992-NA-CT ANGIO ABD WITH PEL-75163/100.000000-Segmentation-46600/1-1.dcm"
file_path = "C:/Users/ariaa/Documents/Jobb/PhD søknad/UiO task/Python/manifest-1669817128730/Colorectal-Liver-Metastases/CRLM-CT-1002/07-12-1992-NA-CT ANGIO ABD WITH CH AND PEL-46457/100.000000-Segmentation-31097/1-1.dcm"

# Load the DICOM file
ds = pydicom.dcmread(file_path)
# Extract the image data
image_data = ds.pixel_array

# Print the shape of the image data
print("Image shape:", image_data.shape)

# Assuming 'image_data' is your numpy array with shape (439, 512, 512)
num_frames = image_data.shape[0]

# Initialize a figure for plotting
fig, ax = plt.subplots()

# Create an empty image plot
im = ax.imshow(image_data[0], cmap='gray')

def update(frame):
    # Update the image data for each frame
    im.set_data(image_data[frame])
    return im,

# Create an animation
anim = FuncAnimation(fig, update, frames=num_frames, interval=10, repeat=False, blit=True)

# Display the animation
plt.show()

file_path = "C:/Users/ariaa/Documents/Jobb/PhD søknad/UiO task/Python/manifest-1669817128730/Colorectal-Liver-Metastases/CRLM-CT-1003/09-24-1994-NA-CT ANGIO LIVER WITH CHPEL-87341/100.000000-Segmentation-19177/1-1.dcm"
# file_path = "C:/Users/ariaa/Documents/Jobb/PhD søknad/UiO task/Python/manifest-1669817128730/Colorectal-Liver-Metastases/CRLM-CT-1001/06-06-1992-NA-CT ANGIO ABD WITH PEL-75163/100.000000-Segmentation-46600/1-1.dcm"

# Load the DICOM file
ds = pydicom.dcmread(file_path)
# Extract the image data
image_data = ds.pixel_array

# Print the shape of the image data
print("Image shape:", image_data.shape)'''

#%% pad missing images as zeros
im_height = len(subject_data_list[0][0])
im_width = len(subject_data_list[0][0][0])

l = []
for i in range(len(subject_data_list)):
    l.append(len(subject_data_list[i]))
max_sequence = max(l)

blank_im_list = [[0]*im_width for _ in range(im_height)]

for i in range(len(subject_data_list)):
    if len(subject_data_list[i]) < max_sequence:
        for j in range(max_sequence - len(subject_data_list[i])):
            subject_data_list[i].append(blank_im_list)
#%% Reduce size of dataset to increase speed

# Remove half of the sequences
subject_data_list_lite = [sequence[:120] for sequence in subject_data_list]

# Reduce the size of each image from 512x512 to 256x256
subject_data_list_lite = [[[row[::2] for row in image[::2]] for image in sequence] for sequence in subject_data_list_lite]

# Print the shape of the reduced dataset
print(len(subject_data_list_lite))
print(len(subject_data_list_lite[0]))
print(len(subject_data_list_lite[0][0]))
print(len(subject_data_list_lite[0][0][0]))

#%% Convert the list of image data to a numpy array

subject_data_array = np.array(subject_data_list_lite) # This gives an error if full dataset is used (cant allocate 46.2 GiB) 

#%% save to file

np.save('image_data.npy', subject_data_array)

#%% test load from file

subject_data = np.load('image_data.npy')

print("Image data array shape:", subject_data.shape)


# %% Save the list of image data to a file using pickle - list uses full data set size

# # Open the file in write mode
# with open("list_data.txt", "wb") as file:
#     # Use the pickle.dump() function to save the list to the file
#     pickle.dump(subject_data_list, file)
# # %% Load the list of image data from a file using pickle

# with open("list_data.txt", "rb") as file:
#     # Use the pickle.load() function to load the list from the file
#     subject_data_list_test = pickle.load(file)

# # %%
# print("Image data array shape:", len(subject_data_list_test), len(subject_data_list_test[0]), len(subject_data_list_test[1]))
