 
# from math import log10, sqrt 
# import cv2 
# import numpy as np 
  
# def PSNR(original, compressed): 
#     mse = np.mean((original - compressed) ** 2) 
#     if(mse == 0):  # MSE is zero means no noise is present in the signal . 
#                   # Therefore PSNR have no importance. 
#         return 100
#     max_pixel = 255.0
#     psnr = 20 * log10(max_pixel / sqrt(mse)) 
#     return psnr 
  
# def main(): 
#      original = cv2.imread("01.jpg")
#      compressed = cv2.imread("after denoise/01.jpg", 1) 
#      value = PSNR(original, compressed) 
#      print(f"PSNR value is {value} dB") 
       
# if __name__ == "__main__": 
#     main() 
from math import log10, sqrt 
import cv2 
import numpy as np 
  
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if mse == 0:  
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
  
def main(): 
    original = cv2.imread("01.jpg")
    compressed = cv2.imread("after denoise/01.jpg", 1)
    
    # Resize compressed image to match the dimensions of the original image
    compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    
    value = PSNR(original, compressed) 
    print(f"PSNR value is {value} dB") 
       
if __name__ == "__main__": 
    main()
