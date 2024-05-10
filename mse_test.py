import cv2
import numpy as np
import os

def mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def calculate_mse_for_folder(folder1, folder2):
    images1 = os.listdir(folder1)
    images2 = os.listdir(folder2)

    for image_name in images1:
        if image_name in images2:
            image1_path = os.path.join(folder1, image_name)
            image2_path = os.path.join(folder2, image_name)

            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)

            if image1.shape[:2] != image2.shape[:2]:  # Resize if dimensions don't match
                image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

            error = mse(image1, image2)
            print(f"Mean Squared Error (MSE) for {image_name}: {error}")
        else:
            print(f"No corresponding image found for {image_name}. Skipping...")

def main():
    folder1 = "data/test_data"
    folder2 = "output_images"
    calculate_mse_for_folder(folder1, folder2)

if __name__ == "__main__":
    main()
