import os
import glob
import time
import torch
import torchvision
from modeling import model
from option import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # For GPU only
device = get_device()

class Tester(): 
    def __init__(self):
        self.scale_factor = 12
        self.net = model.enhance_net_nopool(self.scale_factor, conv_type='dsc').to(device)
        self.net.load_state_dict(torch.load(args.weight_dir, map_location=device))

    def inference(self, image_path):
        print(f"Processing image: {image_path}")
        
        # Read image from path
        data_lowlight = image_from_path(image_path)

        # Scale image to have the resolution of multiple of 4
        data_lowlight = scale_image(data_lowlight, self.scale_factor, device) if self.scale_factor != 1 else data_lowlight

        # Run model inference
        start = time.time()
        enhanced_image, params_maps = self.net(data_lowlight)
        end_time = (time.time() - start)
        
        # Ensure the enhanced image is on CPU and in range [0, 1]
        enhanced_image = enhanced_image.cpu().clamp(0, 1)

        # Print shape and type of the image tensor
        print(f"Enhanced image shape: {enhanced_image.shape}, dtype: {enhanced_image.dtype}")

        # Load result directory and save image
        result_path = os.path.join(args.test_dir, os.path.relpath(image_path, args.input_dir))
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        
        print(f"Saving result to: {result_path}")

        # Save the image using torchvision
        torchvision.utils.save_image(enhanced_image, result_path)
        
        return end_time

    def test(self):
        self.net.eval()
        file_list = glob.glob(os.path.join(args.input_dir, '*'))  # get all the images in all the folders
        sum_time = 0

        for image in file_list:
            sum_time += self.inference(image)

        print(f"Total processing time: {sum_time}")
        print("Test finished!")

if __name__ == '__main__':
    t = Tester()
    t.test()
