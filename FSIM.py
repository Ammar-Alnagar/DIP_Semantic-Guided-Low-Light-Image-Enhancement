import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = {}
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                if img is not None:
                    relative_path = os.path.relpath(img_path, folder)
                    images[relative_path] = img
    return images



def resize_images(images):
    # Determine common dimensions among all images
    dimensions = set()
    for img in images.values():
        dimensions.add(img.shape[:2])
    common_height, common_width = max(dimensions)

    # Resize images to common dimensions
    resized_images = {}
    for path, img in images.items():
        if img.shape[:2] != (common_height, common_width):
            img = cv2.resize(img, (common_width, common_height))
        resized_images[path] = img
    return resized_images

def fsim(img1, img2):
    # Convert images to floating point representation
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Constants for parameter initialization
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Compute gradient magnitude and phase for both images
    Gx1, Gy1 = cv2.spatialGradient(img1)
    Gx2, Gy2 = cv2.spatialGradient(img2)
    grad_mag1 = cv2.magnitude(Gx1, Gy1)
    grad_mag2 = cv2.magnitude(Gx2, Gy2)
    grad_phi1 = cv2.phase(Gx1, Gy1, angleInDegrees=True)
    grad_phi2 = cv2.phase(Gx2, Gy2, angleInDegrees=True)

    # Compute local phase coherence
    sigma_phi = 1.5
    local_phase_coherence = np.exp(-((grad_phi1 - grad_phi2) ** 2) / (2 * (sigma_phi ** 2)))

    # Compute local structural similarity (LSSIM)
    window_size = 11
    k = [0.05, 0.05]
    LSSIM = (2 * grad_mag1 * grad_mag2 + k[0]) / (grad_mag1 ** 2 + grad_mag2 ** 2 + k[0]) * (2 * local_phase_coherence + k[1])

    # Calculate FSIM
    fsim_value = np.mean(LSSIM)

    return fsim_value

def process_image_pair(images1, images2, relative_path):
    if relative_path in images2:
        img1 = images1[relative_path]
        img2 = images2[relative_path]
        
        # Resize images to match dimensions
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        fsim_score = fsim(img1, img2)
        return (relative_path, fsim_score)
    else:
        print(f'No corresponding image for {relative_path} in the second folder')
        return None

def compare_folders(folder1, folder2):
    images1 = load_images_from_folder(folder1)
    images2 = load_images_from_folder(folder2)
    
    # Resize images to ensure consistent dimensions
    resized_images1 = resize_images(images1)
    resized_images2 = resize_images(images2)
    
    results = []
    for relative_path in resized_images1:
        result = process_image_pair(resized_images1, resized_images2, relative_path)
        if result:
            results.append(result)
    
    return results

if __name__ == "__main__":
    folder1 = 'data/test_data'
    folder2 = 'FInal'
    
    fsim_results = compare_folders(folder1, folder2)
    
    for relative_path, score in fsim_results:
        print(f'FSIM for {relative_path}: {score:.4f}')
