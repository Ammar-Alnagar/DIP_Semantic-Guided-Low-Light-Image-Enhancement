import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = {}
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    relative_path = os.path.relpath(img_path, folder)
                    images[relative_path] = img
    return images

def calculate_vsnr(imageA, imageB):
    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Resize images to match dimensions
    if grayA.shape != grayB.shape:
        grayB = cv2.resize(grayB, (grayA.shape[1], grayA.shape[0]))

    # Calculate signal and noise
    signal = np.mean(grayA)
    noise = np.mean(np.abs(grayA - grayB))

    # Calculate VSNR
    vsnr = 20 * np.log10(signal / noise)
    return vsnr

def process_image_pair(images1, images2, relative_path):
    if relative_path in images2:
        imageA = images1[relative_path]
        imageB = images2[relative_path]
        vsnr_score = calculate_vsnr(imageA, imageB)
        return (relative_path, vsnr_score)
    else:
        print(f'No corresponding image for {relative_path} in the second folder')
        return None

def compare_folders(folder1, folder2):
    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)
    
    results = []
    for relative_path in images1:
        result = process_image_pair(images1, images2, relative_path)
        if result:
            results.append(result)
    
    return results

if __name__ == "__main__":
    folder1 = 'data/test_data'
    folder2 = 'FInal'
    
    vsnr_results = compare_folders(folder1, folder2)
    
    for relative_path, score in vsnr_results:
        print(f'VSNR for {relative_path}: {score:.4f} dB')
