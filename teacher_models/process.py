import os
import shutil


root_dir = "./Tiny/ConvNet-4/"
for i in range(400):
    ckpt_path = os.path.join(root_dir, f"checkpoint_{i}.pt")
    if os.path.exists(ckpt_path):
        new_path = os.path.join(root_dir, f"ckpt_{i+1}.pt")
        print(f"Moving {ckpt_path} to {new_path}")
        shutil.move(ckpt_path, new_path)
