import cv2
import os

def remove_noise_from_folder(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Load the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Apply Non-Local Means Denoising to colored image
            denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 20, 20, 7, 21)

            # Save the denoised image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, denoised_image)

            print(f'Denoised image saved: {output_path}')

# Input and output folder names
input_folder = 'output_images'
output_folder = 'denoised_images'

# Call the function to remove noise from images in the input folder and save denoised images to the output folder
remove_noise_from_folder(input_folder, output_folder)
