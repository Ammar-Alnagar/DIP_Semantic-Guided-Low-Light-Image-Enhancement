import cv2
import os

def denoise_image(image):
    # Perform Non-Local Means Denoising
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return denoised_image

# Folder containing images
input_folder = 'output_images'
output_folder = 'after_denoise'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Load the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Apply denoising
        denoised_image = denoise_image(image)

        # Save the denoised image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, denoised_image)

        print(f'Denoised image saved: {output_path}')
