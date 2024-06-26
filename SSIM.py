import os
import cv2
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def resize_image(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]))

def compute_ssim_pair(imageA, imageB):
    if imageA.shape != imageB.shape:
        imageB = resize_image(imageB, imageA.shape)
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

def process_image_pair(images1, images2, relative_path):
    if relative_path in images2:
        imageA = images1[relative_path]
        imageB = images2[relative_path]
        ssim_score = compute_ssim_pair(imageA, imageB)
        return (relative_path, ssim_score)
    else:
        print(f'No corresponding image for {relative_path} in the second folder')
        return None

def compare_folders(folder1, folder2):
    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)
    
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image_pair, images1, images2, relative_path) for relative_path in images1]
        for future in as_completed(futures):
            result = future.result()
            if result and result[1] >= 0.5:  # Filter results with SSIM below 0.5
                results.append(result)
    
    return results

if __name__ == "__main__":
    folder1 = 'data/test_data'
    folder2 = 'FInal'
    
    ssim_results = compare_folders(folder1, folder2)
    
    for relative_path, score in ssim_results:
        print(f'SSIM for {relative_path}: {score:.4f}')
