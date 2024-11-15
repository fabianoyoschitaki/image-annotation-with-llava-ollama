import ollama
from ollama import generate

import glob
import pandas as pd
from PIL import Image

import os
from io import BytesIO

import time
import re

model = 'llava:13b-v1.6'
image_descriptions_file_name = 'image_descriptions_' + ''.join(re.findall(r'[a-zA-Z]', model)) + '.csv'

# Load the DataFrame from a CSV file, or create a new one if the file doesn't exist
def load_or_create_dataframe(filename):
    if os.path.isfile(filename):
        print(f"Image description file exists: {filename}")
        df = pd.read_csv(filename)
    else:
        print(f"Image description file doesn't exist: {filename}")
        df = pd.DataFrame(columns=['image_file', 'description'])
    return df

df = load_or_create_dataframe(image_descriptions_file_name)

def get_png_files(folder_path):
    return glob.glob(f"{folder_path}/*.png")

# get the list of image files in the folder yopu want to process
image_files = get_png_files("./images") 
image_files.sort()

print(image_files[:3])
print(df.head())


# processing the images 
def process_image(image_file):
    start_time = time.time()
    print(f"\n\n----------------------------------------\nProcessing {image_file} with model {model}\n")
    with Image.open(image_file) as img:
        with BytesIO() as buffer:
            img.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

    full_response = ''
    # Generate a description of the image
    for response in generate(
        # model='llava:13b-v1.6', 
        model='llama3.2-vision:latest',
        prompt='describe this image and make sure to include anything notable about it (include text you see in the image):', 
        images=[image_bytes], 
        stream=True):
        # Print the response to the console and add it to the full response
        print(response['response'], end='', flush=True)
        full_response += response['response']

    # Add a new row to the DataFrame
    df.loc[len(df)] = [image_file, full_response]
    total_time = (time.time() - start_time) * 1000
    print(f"\n\nTotal time to process {image_file} with model {model}: {total_time:.2f} ms")


for image_file in image_files:
    if image_file not in df['image_file'].values:
        process_image(image_file)

# Save the DataFrame to a CSV file
df.to_csv(image_descriptions_file_name, index=False)
print(f"Image description file saved: {image_descriptions_file_name}")

