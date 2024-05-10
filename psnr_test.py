from math import log10, sqrt 
import cv2 
import numpy as np 
import os

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if mse == 0:  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def calculate_PSNR_for_folder(original_folder, compressed_folder):
    original_images = os.listdir(original_folder)
    compressed_images = os.listdir(compressed_folder)

    psnr_values = []

    for original_image_name in original_images:
        if original_image_name in compressed_images:
            original_path = os.path.join(original_folder, original_image_name)
            compressed_path = os.path.join(compressed_folder, original_image_name)

            original = cv2.imread(original_path)
            compressed = cv2.imread(compressed_path, 1)

            if original.shape[:2] != compressed.shape[:2]:  # Resize compressed image to match dimensions of original image
                compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))

            value = PSNR(original, compressed) 
            print(f"PSNR value for {original_image_name} is {value} dB") 

            psnr_values.append(value)

    # Calculate the average PSNR value
    average_psnr = np.mean(psnr_values)
    print(f"Average PSNR value: {average_psnr} dB")

def main(): 
    original_folder = "data/test_data"
    compressed_folder = "output_images"
    calculate_PSNR_for_folder(original_folder, compressed_folder)
       
if __name__ == "__main__": 
    main()
