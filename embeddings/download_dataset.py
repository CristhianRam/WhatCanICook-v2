import os

import kagglehub

output_folder = os.path.join(os.curdir, "dataset")

# Create the folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Download latest version
path = kagglehub.dataset_download(
    "realalexanderwei/food-com-recipes-with-ingredients-and-tags",
    force_download=True,
    output_dir=output_folder,
)

print("Path to dataset files:", path)
