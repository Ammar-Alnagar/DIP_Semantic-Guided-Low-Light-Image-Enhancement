import os
from PIL import Image
import torch
import piq
import numpy as np




# Folder containing images
folder_path = "after_HE"  

# List to store BRISQUE scores
brisque_scores = []

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is an image (you may need to adjust this condition based on your file naming conventions)
    if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        # Load the image
        image_path = os.path.join(folder_path, file_name)
        image = Image.open(image_path)

        # Convert image to tensor
        image_tensor = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        # Calculate BRISQUE score
        score = piq.brisque(image_tensor.unsqueeze(0))

        # Append the score to the list
        brisque_scores.append(score.item())

        print(f"Image: {file_name}, BRISQUE Score: {score.item()}")

# Calculate the average BRISQUE score
average_brisque_score = np.mean(brisque_scores)
print(f"Average BRISQUE Score: {average_brisque_score}")
