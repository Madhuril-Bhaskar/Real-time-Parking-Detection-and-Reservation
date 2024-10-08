################################################################################
#               Real-Time parking detecttion and reservation                   #                  
#                                                                              #
#                         Data Preprocessing                                   #
#                                                                              #
################################################################################





################################################################################
#    Loading Libraries        
################################################################################

from email.mime import image
import pandas as pd
import os
import PIL
import shutil
import xml.etree.ElementTree as ET
import random
from PIL import Image
from sklearn.semi_supervised import LabelSpreading
import matplotlib.pyplot as plt


################################################################################
#    Merging all folders in one      
################################################################################

source_path = [
    "E:/PKLot/PKLot/PKLot/PUCPR/Cloudy",
    "E:/PKLot/PKLot/PKLot/PUCPR/Rainy",
    "E:/PKLot/PKLot/PKLot/PUCPR/Sunny",
    "E:/PKLot/PKLot/PKLot/UFPR04/Cloudy",
    "E:/PKLot/PKLot/PKLot/UFPR04/Rainy",
    "E:/PKLot/PKLot/PKLot/UFPR04/Sunny",
    "E:/PKLot/PKLot/PKLot/UFPR05/Cloudy",
    "E:/PKLot/PKLot/PKLot/UFPR05/Rainy",
    "E:/PKLot/PKLot/PKLot/UFPR05/Sunny"
]

destination_path = "data/total_content"

if not os.path.exists(destination_path) :
    os.makedirs(destination_path)

# Iterate through source folder and merge contents

for source_folder in source_path: 
    for root, _, files in os.walk(source_folder) :
        for file in files :
            source_file_path      = os.path.join(root, file)
            destination_file_path = os.path.join(destination_path, file)
            shutil.copy(source_file_path, destination_file_path)
        

################################################################################
#    Modify labels into format : class x_center y_center width height  
################################################################################

input_path  = "E:/PKLot/PKLot/data/total_content"
output_path = "E:/PKLot/PKLot/data/labels-xml"

image_width  = 1280
image_height = 720

class_mapping = {"1" : 1 ,"0" : 0}

if not os.path.exists(output_path) :
    os.makedirs(output_path)

# Extract only xml files from total_contents

# All files name in input path with .xml extension
xml_files = [f for f in os.listdir(input_path) if f.endswith(".xml")]  

for xml_file in xml_files :
    xml_path = os.path.join(input_path, xml_file)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Split from the extension gives the first part 
    txt_filename = os.path.splitext(xml_file)[0] + ".txt"  
    txt_path     = os.path.join(output_path, txt_filename)
    
    # Reading data from xml files adn write into text file
    with open(txt_path, "w") as txt_file :
        for space in root.findall("space") :
            occupied = space.get("occupied")
            class_id = class_mapping.get(occupied, -1) # type: ignore
            
            if class_id == -1 :
                continue
            
            rotated_rect    = space.find("rotatedRect")
            center          = rotated_rect.find("center") # type: ignore
            size            = rotated_rect.find("size")   # type: ignore
            
            center_x = float(center.get("x")) # type: ignore
            center_y = float(center.get("y")) # type: ignore
            width    = float(size.get("w"))   # type: ignore
            height   = float(size.get("h"))   # type: ignore
            
            # Normalizing 
            x_center = center_x / image_width
            y_center = center_y / image_height
            w        = width / image_width
            h        = height / image_height
            
            # Writting
            txt_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    new_xml_path = os.path.join(output_path, xml_file)
    os.rename(xml_path, new_xml_path)
 
    
################################################################################
#    Move the txt files from xml folder to the total_content folder
################################################################################

source_folder      = "E:/PKLot/PKLot/data/labels-xml"
destination_folder = "E:/PKLot/PKLot/data/total_content"

# All files name with .txt extension in source_path
txt_files = [f for f in os.listdir(source_folder) if f.endswith(".txt")]

for txt_file in txt_files :
    source_path = os.path.join(source_folder, txt_file)
    destination_path = os.path.join(destination_folder, txt_file)
    shutil.move(source_path, destination_path)
    
    
################################################################################
#    Train_test_val folders split
################################################################################

source_folder   = "E:/PKLot/PKLot/data/total_content"
train_folder    = "E:/PKLot/PKLot/data/train"
test_folder     = "E:/PKLot/PKLot/data/test"
val_folder      = "E:/PKLot/PKLot/data/val"

train_ratio = 0.7
test_ratio  = 0.15
val_ratio   = 0.15

for folder in [train_folder, test_folder, val_folder] :
    if not os.path.exists(folder) :
        os.makedirs(folder)
        
all_files = os.listdir(source_folder)
image_files = [f for f in all_files if f.endswith(".jpg")]


# Calculate the number of samples for each split
num_samples = len(image_files)
num_train = int(train_ratio * num_samples)
num_test = int(test_ratio * num_samples)
num_val = num_samples - num_test - num_train

random.shuffle(image_files)

# Picking up the indexes
train_files  = image_files[ : num_train]
test_files   = image_files[num_train : num_train + num_test]
val_files    = image_files[num_train+num_test:]

# Move corresponding txt files along with images

# for test folder :
for file in test_files :
    # Move images
    source_image_path       = os.path.join(source_folder, file)
    destination_image_path  = os.path.join(test_folder, file)
    shutil.move(source_image_path, destination_image_path)
    
    # Move txt files
    txt_filename         = os.path.splitext(file)[0] + ".txt"
    source_txt_path      = os.path.join(source_folder, txt_filename)
    destination_txt_path = os.path.join(test_folder, txt_filename)
    shutil.move(source_txt_path, destination_txt_path)

# for val folder : 
for file in val_files :
    # Move images
    source_image_path       = os.path.join(source_folder, file)
    destination_image_path  = os.path.join(val_folder, file)
    shutil.move(source_image_path, destination_image_path)
    
    # Move txt files
    txt_filename         = os.path.splitext(file)[0] + ".txt"
    source_txt_path      = os.path.join(source_folder, txt_filename)
    destination_txt_path = os.path.join(val_folder, txt_filename)
    shutil.move(source_txt_path, destination_txt_path)       
        
# for train folder : 
for file in train_files :
    # Move images
    source_image_path       = os.path.join(source_folder, file)
    destination_image_path  = os.path.join(train_folder, file)
    shutil.move(source_image_path, destination_image_path)
    
    # Move txt files
    txt_filename         = os.path.splitext(file)[0] + ".txt"
    source_txt_path      = os.path.join(source_folder, txt_filename)
    destination_txt_path = os.path.join(train_folder, txt_filename)
    shutil.move(source_txt_path, destination_txt_path)  
    
 
################################################################################
#    Organize the datasets into the following structure :
#    data
#    |--train
#    |     |--images
#    |     |--Label
#    |--val
#    |     |--images
#    |     |--labels
#    |--test
#    |     |--images
#    |     |--labels
#
################################################################################
   

datasets      = ['train', 'val', 'test']
source_folder = "E:/PKLot/PKLot/data"

for dataset in datasets:
    dataset_folder = os.path.join(source_folder, dataset)
    
    # creating images and labels folders
    images_folder = os.path.join(dataset_folder, "images")
    labels_folder = os.path.join(dataset_folder, "labels")
    
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)
    
    # Organize image and labels files
    for file in os.listdir(dataset_folder) :
        if(file.endswith(".jpg")) :
            image_path        = os.path.join(dataset_folder, file)
            image_destination = os.path.join(images_folder, file)
            shutil.move(image_path, image_destination)
        elif file.endswith(".txt") :
            label_path        = os.path.join(dataset_folder, file)
            label_destination = os.path.join(labels_folder, file)
            shutil.move(label_path, label_destination)


################################################################################
#    Dataset exploration 
################################################################################

dataset_path = "E:/PKLot/PKLot/data"

# Set paths for training, test and validation image sets

train_image_path = os.path.join(dataset_path, 'train', 'images')
test_image_path  = os.path.join(dataset_path, 'test', 'images')
val_image_path   = os.path.join(dataset_path, 'val', 'images')

# Initialize sets to hold the unique sizes of images
train_image_sizes = set()
test_image_sizes  = set()
val_image_sizes   = set()

# Check train Images sizes
for filename in os.listdir(train_image_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(train_image_path, filename)
        with Image.open(image_path) as img:
            train_image_sizes.add(img.size)

# Check test Images sizes
for filename in os.listdir(test_image_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(test_image_path, filename)
        with Image.open(image_path) as img:
            test_image_sizes.add(img.size)

# Check train Images sizes
for filename in os.listdir(val_image_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(val_image_path, filename)
        with Image.open(image_path) as img:
            val_image_sizes.add(img.size)

# Check if all images in training set have the same size
if len(train_image_sizes) == 1:
    print(f"All training images have the same size: {train_image_sizes.pop()}")
else:
    print("Training images have varying sizes.")

# Check if all images in training set have the same size
if len(test_image_sizes) == 1:
    print(f"All training images have the same size: {test_image_sizes.pop()}")
else:
    print("Training images have varying sizes.")

# Check if all images in validation set have the same size
if len(val_image_sizes) == 1:
    print(f"All validation images have the same size: {val_image_sizes.pop()}")
else:
    print("Validation images have varying sizes.")


# List all jpg images in the train_set

image_files = [f for f in os.listdir(train_image_path) if f.endswith('.jpg')]

# Select 8 images at equal intervals
num_images = len(image_files)
selected_images = [image_files[i] for i in range(0, num_images, num_images // 8)]

# Create a 2x4 subplot
fig, axes = plt.subplots(2, 4, figsize = (20, 11))

# Display each of the selected images

for ax, image_file in zip(axes.ravel(), selected_images) :
    image_path = os.path.join(train_image_path, image_file)
    image = Image.open(image_path)
    ax.imshow(image)
    ax.axis('off')
    
plt.suptitle("Sample Images from training set", fontsize = 40)
plt.tight_layout()
plt.show()


