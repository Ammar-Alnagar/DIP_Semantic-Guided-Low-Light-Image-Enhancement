import os
import cv2
import numpy as np

def multi_scale_retinex(image, sigma_list):
    retinex = np.zeros_like(image, dtype=np.float32)
    for sigma in sigma_list:
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        retinex += np.log10(image.astype(np.float32) + 1e-6) - np.log10(blurred + 1e-6)
    retinex /= len(sigma_list)
    return retinex

def MSR_enhancement(image, sigma_list, gain, offset):
    retinex = multi_scale_retinex(image, sigma_list)
    enhanced_image = np.clip(image * gain * (retinex + offset), 0, 255).astype(np.uint8)
    return enhanced_image

input_folder = "test/after denoise"
output_folder = "test/aftermsr"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Parameters for the Multi-Scale Retinex
sigma_list = [15, 80, 250]
gain = 128
offset = 128

# Process images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the low-light image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Enhance the low-light image
        enhanced_image = MSR_enhancement(image, sigma_list, gain, offset)

        # Save the enhanced image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, enhanced_image)

        print(f"Processed {filename}")

print("All images processed.")
