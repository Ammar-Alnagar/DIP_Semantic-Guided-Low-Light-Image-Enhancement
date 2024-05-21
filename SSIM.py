from skimage.metrics import structural_similarity as ssim
import cv2
import os

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

def main():
    # image_folder1 = "test_output"
    image_folder1 = "data/test_data"
    image_folder2 = "after_HE"

    image_files1 = os.listdir(image_folder1)
    image_files2 = os.listdir(image_folder2)

    ssim_values = []

    for file1 in image_files1:
        if file1 in image_files2:
            image_path1 = os.path.join(image_folder1, file1)
            image_path2 = os.path.join(image_folder2, file1)
            ssim_index = calculate_ssim(image_path1, image_path2)
            print(f"SSIM for {file1}: {ssim_index}")
            ssim_values.append(ssim_index)
        else:
            print(f"No corresponding file found for {file1}")

    # Calculate the average SSIM value
    average_ssim = sum(ssim_values*100) / len(ssim_values)
    print(f"Average SSIM: {average_ssim}")

if __name__ == "__main__":
    main()
