import cv2
import os

def run_histogram_equalization(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over all files in the input folder
    for filename in os.listdir(input_folder):
        # Construct the full file path
        input_path = os.path.join(input_folder, filename)
        
        # Check if the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            rgb_img = cv2.imread(input_path)
            
            # Check if the image is read correctly
            if rgb_img is None:
                print(f"Error reading {input_path}. Skipping.")
                continue

            # Convert from RGB color-space to YCrCb
            ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

            # Equalize the histogram of the Y channel
            ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

            # Convert back to RGB color-space from YCrCb
            equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

            # Construct the output file path
            output_path = os.path.join(output_folder, filename)

            # Save the equalized image
            cv2.imwrite(output_path, equalized_img)
            print(f"Processed {input_path} and saved to {output_path}")

# Specify the input and output folder paths
input_folder = 'output_images'
output_folder = 'after_HE'

# Run the function
run_histogram_equalization(input_folder, output_folder)
