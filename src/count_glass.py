import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN
from skimage.feature import blob_dog

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    return thresh_image

def detect_glasses(image):
    # Use Difference of Gaussians to detect blobs which could be glasses
    blobs = blob_dog(image, min_sigma=2, max_sigma=30, threshold=0.1)
    return len(blobs)

# Process each image in your folder
folder_path = '/opt/project/dataset/test/'
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    processed_image = preprocess_image(image_path)
    glass_count = np.rint(detect_glasses(processed_image)/50)
    print(f'Number of glasses in {image_file}: {glass_count}')
