import cv2

def print_pixel_values(image):
    # Iterate through each pixel and print its value
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            print(f"Pixel at ({i}, {j}): {image[i, j]}")

# Load the image
image1 = cv2.imread('01.jpg')


# Print pixel values

print_pixel_values(image1) 


