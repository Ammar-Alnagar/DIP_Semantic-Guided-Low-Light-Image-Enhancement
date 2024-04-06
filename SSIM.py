from skimage.metrics import structural_similarity as ssim
import cv2

# Load two images
image1 = cv2.imread('01.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('after denoise/01.jpg', cv2.IMREAD_GRAYSCALE)

# Resize images to the same dimensions if they are not already the same
if image1.shape != image2.shape:
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Calculate SSIM
ssim_index = ssim(image1, image2)

# Print SSIM index
print("Structural Similarity Index (SSIM):", ssim_index)
