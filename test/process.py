import os
import cv2

def exposure_enhancer(image, alpha, beta):
    # Apply exposure adjustment using alpha and beta parameters
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Normalize pixel values to the range [0, 255]
    adjusted_image = cv2.normalize(adjusted_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return adjusted_image

def process_images(folder_path, output_folder, alpha, beta):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the input image
            image_path = os.path.join(folder_path, filename)
            input_image = cv2.imread(image_path)
            
            # Apply exposure enhancement
            enhanced_image = exposure_enhancer(input_image, alpha, beta)
            
            # Write the enhanced image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, enhanced_image)
            print(f"Processed: {filename}")

# Define the folder containing the input images
input_folder = 'input_images/'

# Define the folder where the processed images will be saved
output_folder = 'output_images/'

# Set the alpha and beta parameters for exposure adjustment
alpha = 1.0  # Multiply the pixel values by 1.2
beta = 0  # Add 10 to all pixel values

# Process the images in the input folder
process_images(input_folder, output_folder, alpha, beta)
