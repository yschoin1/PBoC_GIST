# Small project during Physical Biology of the Cell(PBoC) at Gwangju Institute
# Science and Technology(GIST).
# In this project, I analyzed 2 freely diffusing bacteria and 1 bacteria that is
# being chased by a neutrophil.
# The first bacteria movie was obtained at GIST by diluting E. coli in growth media.
# The second and third bacteria were taken from David Rogers (Vanderbilt Univ.)
# 1950 neutrophil chase movie.
# Coded by Yongseok Choi at GIST and advised by Griffin Chure at Caltech.

#Import necessary tools
import glob
import os
import warnings

# Our numerical workhorses
import numpy as np
import pandas as pd

# BE/Bi 103 utilities
import bebi103

# Image processing tools
import skimage
import skimage.io
import skimage.morphology
import skimage.segmentation

# Import plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2,
      'axes.labelsize': 18,
      'axes.titlesize': 18,
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style('darkgrid', rc=rc)

# Suppress future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Directory containing the image files
data_dir = "/Users/Rio/Documents/2015-16_winter/PBoC@GIST/Project/free_diffusion/tiffs"

# Information about the first bacteria movie
fps = 7
time = 1/fps # in seconds
inter_pixel_distance = 160 # in nm

# Glob string for first movie images
im_glob = os.path.join(data_dir, 'new_diffusiont*.tif')

# Get list of files in directory
im_list = glob.glob(im_glob)

# Return only the R channel since RGB channels are all the same.
# This function also only selects region of interest (ROI) for bacteria 1.
def squish_rgb_and_show_ROI(fname):
    return skimage.io.imread(fname)[200:300,300:400,0]

# Image collection of only the R channel and ROI for bacteria 1.
ic = skimage.io.ImageCollection(im_glob, conserve_memory=True, \
                                load_func=squish_rgb_and_show_ROI)

centroid = [] # Empty list to store centroid coordinates

# Threshold from frame 100 to 200, label them and get the centroid with
# regionprops.
frame = 200 - 100 + 1
for i in range(99, 200):
    # Decided by looking at the histogram of ic[138]
    threshold = 100
    im_bw_labeled = (skimage.measure.label(ic[i] < threshold))
    props = skimage.measure.regionprops(im_bw_labeled)
    centroid.append(props[0].centroid)

def compute_D(frame, centroid):
    """
    This function takes in total frame number and the centroid and returns the
    short and long diffusion constant (D).
    """
    short_D = [] # Empty list to store short diffusion constant
    long_D = [] # Empty list to store long diffusion constant
    travel_distance = [] # List to save the total travel distance of the particle.

    # Obtain short diffusion constant(D), where D = <d^2> / (4t)
    # where d is distance and t is time
    for i in np.arange(frame - 1):
        distance_squared = ((centroid[i][0] - centroid[i+1][0])**2 \
                            + (centroid[i][1] - centroid[i+1][1])**2) \
                            * (inter_pixel_distance)**2 # in nm^2
        short_D.append(distance_squared / (4*time))
        travel_distance.append(distance_squared)

    # Obtain long time D
    long_D = (sum(travel_distance) / frame) / (4*time)

    return short_D, long_D

# Compute the short and long D for bacteria 1
short_D, long_D = compute_D(frame, centroid)

# Centroid y values are flipped around. Correct this before saving to dataframes
for i in range(len(centroid)):
    centroid[i] = (centroid[i][1], 300 - 200 + 1 - centroid[i][0])

# Append 0 to the end of long_D to store in dataframe
temp = long_D
long_D = np.empty(2)
long_D[0] = temp
long_D[1] = 0

# Change the units of short_D and long_D to µm^2
for i in range(len(short_D)):
    short_D[i] = short_D[i] / 1000000
long_D[0] = long_D[0] / 1000000

# Save the centroid, short_d, long_d data into dataframes
bac1_centroid_df = pd.DataFrame(centroid)
bac1_centroid_df.columns = ['X_cent (pixel)', 'Y_cent (pixel)']
bac1_short_D_df = pd.DataFrame(short_D)
bac1_short_D_df.columns = ['D (µm^2 / sec)']
bac1_long_D_df = pd.DataFrame(long_D)
bac1_long_D_df.columns = ['D (µm^2 / sec)']

# Export dataframes to csv files
bac1_centroid_df.to_csv('~/Desktop/Bac1_centroid.csv', index=False, header=True)
bac1_short_D_df.to_csv('~/Desktop/Bac1_short_D.csv', index=False, header=True)
bac1_long_D_df.to_csv('~/Desktop/Bac1_long_D.csv', index=False, header=True)



# Rogers movie
# Get diffusion constant for a free floating bacteria in the movie from frame 390 to 423

# Directory containing the image files
data_dir = "/Users/Rio/Documents/2015-16_winter/PBoC@GIST/Project/16.9-neutrophil_chase.tiff"

# Information about the microscopy images
fps = 5 # not known but interpolated from the free floating bacteria
time = 1/fps # in seconds

# There is no information about the interpixel distance of the microscope for this video
# However, a normal red blood cell is about 8 µm in diameter.
# Use this information to calculate the inter_pixel distance
inter_pixel_distance = 180 # in nM
frame = 423 - 390 + 1 # Number of frames

# Load the image as an ImageCollection
ic = skimage.io.ImageCollection(data_dir)

im_list = [] # Empty list to store images

# Only get the free floating bacteria and get G channel
for i in range(389, 423):
    im_list.append(ic[i][130:180, 235:270, 1])

# Set threshold and get binary image, label and get centroid with regionprops
threshold = 100 # Obtained by looking at histograms of images
centroid = [] # Empty list to store centroids
im_bw_labeled_removed = [] # Empty list to store labeled images

for i in range(frame):
    im_bw_labeled = skimage.measure.label(im_list[i] < threshold)
    # Remove all the small objects
    im_bw_labeled_removed.append(skimage.morphology.remove_small_objects(im_bw_labeled, min_size=5))
    props = skimage.measure.regionprops(im_bw_labeled_removed[i])
    centroid.append(props[0].centroid)

# Compute the short and long D
short_D, long_D = compute_D(frame, centroid)

# Centroid y values are flipped around. Correct this before saving to dataframes
for i in range(len(centroid)):
    centroid[i] = (centroid[i][1], 180 - 130 + 1 - centroid[i][0])

# Append 0 to the end of long_D to store in dataframe
temp = long_D
long_D = np.empty(2)
long_D[0] = temp
long_D[1] = 0

# Change the units of short_D and long_D to µm^2
for i in range(len(short_D)):
    short_D[i] = short_D[i] / 1000000
long_D[0] = long_D[0] / 1000000

# Save the centroid, short_d, long_d data into dataframes
bac2_centroid_df = pd.DataFrame(centroid)
bac2_centroid_df.columns = ['X_cent (pixel)', 'Y_cent (pixel)']
bac2_short_D_df = pd.DataFrame(short_D)
bac2_short_D_df.columns = ['D (µm^2 / sec)']
bac2_long_D_df = pd.DataFrame(long_D)
bac2_long_D_df.columns = ['D (µm^2 / sec)']

# Export dataframes to csv files
bac2_centroid_df.to_csv('~/Desktop/Bac2_centroid.csv', index=False, header=True)
bac2_short_D_df.to_csv('~/Desktop/Bac2_short_D.csv', index=False, header=True)
bac2_long_D_df.to_csv('~/Desktop/Bac2_long_D.csv', index=False, header=True)


# Get diffusion constant for a bacteria being chased in the movie from frame 257 to 291
# Frames 257 - 291
frame = 291 - 257 + 1
im_list = [] # Empty list to store images

# Only store frames that contain freely moving bacteria
for i in range(256, 291):
    im_list.append(ic[i])

# Slice out the ROI and only the green channel
for i in range(frame):
    im_list[i] = im_list[i][70:170, 0:100, 1]

# Set threshold, clear the border, remove small objects and then label
threshold = 115 # Obtained by looking at histograms of images
im_bw_border_removed_labeled = [] # Empty list to store labeled images

for i in range(frame):
    im_bw_border = skimage.segmentation.clear_border(im_list[i] < threshold)
    # Remove all the small objects
    im_bw_border_removed = skimage.morphology.remove_small_objects(im_bw_border, min_size=53)
    im_bw_border_removed_labeled.append(skimage.measure.label(im_bw_border_removed))

# Apply an area filter
bacteria = [] # Empty list to store binary bacteria image
for i in range(len(im_bw_border_removed_labeled)):
    for j in range(1, im_bw_border_removed_labeled[i].max() + 1):
        props = skimage.measure.regionprops(im_bw_border_removed_labeled[i] == j)
        # 63 was hard coded to eliminate one blob
        if (props[0].area > 50) and (props[0].area < 122) and (props[0].area != 63):
            bacteria.append(im_bw_border_removed_labeled[i] == j)

# Wrong object was selected by the area filter above
# Change the frame 29 with bacteria
bacteria[28] = im_bw_border_removed_labeled[28] == 1
skimage.io.imshow(bacteria[28])

# Let's track this maverick
centroid = [] # Empty list to store centroids
for i in range(frame):
    props = skimage.measure.regionprops(bacteria[i])
    centroid.append(props[0].centroid)

# Compute the short and long D
short_D, long_D = compute_D(frame, centroid)

# Centroid y values are flipped around. Correct this before saving to dataframes
for i in range(len(centroid)):
    centroid[i] = (centroid[i][1], 170 - 70 + 1 - centroid[i][0])

# Append 0 to the end of long_D to store in dataframe
temp = long_D
long_D = np.empty(2)
long_D[0] = temp
long_D[1] = 0

# Change the units of short_D and long_D to µm^2
for i in range(len(short_D)):
    short_D[i] = short_D[i] / 1000000
long_D[0] = long_D[0] / 1000000

# Save the centroid, short_d, long_d data into dataframes
bac3_centroid_df = pd.DataFrame(centroid)
bac3_centroid_df.columns = ['X_cent (pixel)', 'Y_cent (pixel)']
bac3_short_D_df = pd.DataFrame(short_D)
bac3_short_D_df.columns = ['D (µm^2 / sec)']
bac3_long_D_df = pd.DataFrame(long_D)
bac3_long_D_df.columns = ['D (µm^2 / sec)']

# Export dataframes to csv files
bac3_centroid_df.to_csv('~/Desktop/Bac3_centroid.csv', index=False, header=True)
bac3_short_D_df.to_csv('~/Desktop/Bac3_short_D.csv', index=False, header=True)
bac3_long_D_df.to_csv('~/Desktop/Bac3_long_D.csv', index=False, header=True)
