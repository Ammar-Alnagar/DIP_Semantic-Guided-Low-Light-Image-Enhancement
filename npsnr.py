import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 10 * np.log10((max_pixel ** 2) / mse)
    return psnr

def process_image_pair(original_path, compressed_path):
    original = cv2.imread(str(original_path))
    compressed = cv2.imread(str(compressed_path))

    if original is None:
        logging.warning(f"Could not read image {original_path}")
        return None
    if compressed is None:
        logging.warning(f"Could not read image {compressed_path}")
        return None

    if original.shape[:2] != compressed.shape[:2]:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))

    return PSNR(original, compressed)

def calculate_PSNR_for_folder(original_folder, compressed_folder):
    original_folder = Path(original_folder)
    compressed_folder = Path(compressed_folder)

    original_images = list(original_folder.glob('*'))
    compressed_images = {img.name: img for img in compressed_folder.glob('*')}

    psnr_values = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for original_path in original_images:
            compressed_path = compressed_images.get(original_path.name)
            if compressed_path:
                futures.append(executor.submit(process_image_pair, original_path, compressed_path))

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                psnr_values.append(result)
                logging.info(f"PSNR value: {result:.2f} dB")

    # Calculate the average PSNR value
    if psnr_values:
        average_psnr = np.mean(psnr_values)
        logging.info(f"Average PSNR value: {average_psnr:.2f} dB")
    else:
        logging.warning("No PSNR values calculated.")

def main():
    parser = argparse.ArgumentParser(description="Calculate PSNR between two folders of images.")
    parser.add_argument('original_folder', type=str, help="Path to the folder containing the original images.")
    parser.add_argument('compressed_folder', type=str, help="Path to the folder containing the compressed images.")
    args = parser.parse_args()

    calculate_PSNR_for_folder(args.original_folder, args.compressed_folder)

if __name__ == "__main__":
    main()
