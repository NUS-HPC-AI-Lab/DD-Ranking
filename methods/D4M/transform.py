import os
from PIL import Image
import torchvision.transforms as transforms
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_size', default=32, type=int, 
                        help='image size')
    parser.add_argument('--root_dir', default='./distilled_data/tiny_imagenet_ipc50_50_s0.7_g8.0_kmexpand1', type=str, 
                        help='root dir')
    parser.add_argument('--save_dir', default='./resized_data/TinyImageNet/IPC50', type=str, 
                        help='save dir')
    args = parser.parse_args()
    return args

args = parse_args()
root_dir = args.root_dir
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((args.input_size, args.input_size)),  
    transforms.ToTensor(),        
])

for subfolder_name in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder_name)
    
    if os.path.isdir(subfolder_path):
        subfolder_save_path = os.path.join(save_dir, subfolder_name)
        os.makedirs(subfolder_save_path, exist_ok=True)

        for filename in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, filename)
            
            if file_path.endswith('.jpg') or file_path.endswith('.png'):
                try:
                    image = Image.open(file_path)
                    image = transform(image)
                    image = transforms.ToPILImage()(image)
                    save_path = os.path.join(subfolder_save_path, filename)
                    image.save(save_path)

                except Exception as e:
                    print("Error!")

print("Done!")
