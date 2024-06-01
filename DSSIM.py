import os
import cv2
from skimage.metrics import structural_similarity as ssim

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

def calculate_dssim(imageA, imageB):
    # Resize images to match dimensions
    if imageA.shape != imageB.shape:
        imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    dssim_score, _ = ssim(grayA, grayB, full=True)
    return dssim_score

def process_image_pair(images1, images2, relative_path):
    if relative_path in images2:
        imageA = images1[relative_path]
        imageB = images2[relative_path]
        dssim_score = calculate_dssim(imageA, imageB)
        return (relative_path, dssim_score)
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
    
    dssim_results = compare_folders(folder1, folder2)
    
    for relative_path, score in dssim_results:
        print(f'DSSIM for {relative_path}: {score:.4f}')
