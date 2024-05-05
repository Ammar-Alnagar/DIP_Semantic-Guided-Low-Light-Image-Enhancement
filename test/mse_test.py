import cv2
import numpy as np

def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

# Load two images
image1 = cv2.imread('01.jpg')
image2 = cv2.imread('after denoise/01.jpg')

# Resize images to the same dimensions if they are not already the same
image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

# Calculate MSE
error = mse(image1, image2)
print("Mean Squared Error (MSE):", error)
