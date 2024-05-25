from skimage.metrics import structural_similarity as ssim
import cv2
import os
import sys

def calculate_ssim(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Resize images to the same dimensions if they are not already the same
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Calculate SSIM
    ssim_index = ssim(image1, image2)

    return ssim_index

def calculate_average_ssim(folder1, folder2):
    image_files1 = os.listdir(folder1)
    image_files2 = os.listdir(folder2)

    ssim_values = []

    for file1 in image_files1:
        if file1 in image_files2:
            image_path1 = os.path.join(folder1, file1)
            image_path2 = os.path.join(folder2, file1)
            ssim_index = calculate_ssim(image_path1, image_path2)
            ssim_percentage = ssim_index * 100  # Convert SSIM to percentage
            print(f"SSIM for {file1}: {ssim_percentage:.2f}%")
            ssim_values.append(ssim_index)
        else:
            print(f"No corresponding file found for {file1}")

    # Calculate the average SSIM value
    if ssim_values:
        average_ssim = sum(ssim_values) / len(ssim_values)
        average_ssim_percentage = average_ssim * 100  # Convert average SSIM to percentage
        print(f"Average SSIM: {average_ssim_percentage:.2f}%")
    else:
        print("No SSIM values calculated.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py folder1 folder2")
        return

    folder1 = sys.argv[1]
    folder2 = sys.argv[2]

    calculate_average_ssim(folder1, folder2)

if __name__ == "__main__":
    main()
