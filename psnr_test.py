


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
  
def calculate_psnr(original_path, compressed_path):
    original = cv2.imread(original_path)
    compressed = cv2.imread(compressed_path, 1)
    
    # Resize compressed image to match the dimensions of the original image
    compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    
    value = PSNR(original, compressed) 
    return value

def main(): 
    original_folder = "data/test_data"
    # original_folder = "test_output"
    compressed_folder = "test/after denoise"
    
    original_files = os.listdir(original_folder)
    
    for file in original_files:
        original_path = os.path.join(original_folder, file)
        compressed_path = os.path.join(compressed_folder, file)
        
        if os.path.isfile(compressed_path):
            value = calculate_psnr(original_path, compressed_path)
            print(f"PSNR value for {file} is {value} dB") 
        else:
            print(f"No compressed file found for {file}")

if __name__ == "__main__": 
    main()
