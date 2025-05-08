import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Unused or unrecognized kwargs: padding.")

import traceback
import shutil
import numpy as np
import cv2
import os
import json
import heapq
import cv2
import matplotlib.pyplot as plt
from torchmetrics.functional.multimodal import clip_score
import torch
from functools import partial
from PIL import Image
from tqdm import tqdm
import re
import glob
# from multiprocessing import Pool, Manager
# from tqdm.contrib.concurrent import process_map


def total_variation_loss(image):
    """
    Calculate the Total Variation Loss for an image.

    Parameters:
    image (numpy.ndarray): The input image.

    Returns:
    float: The total variation loss of the image.
    """
    # Calculate the difference of pixel values between adjacent pixels
    image= image.astype(np.int64)
    pixel_diff_x = np.diff(image, axis=0)
    pixel_diff_y = np.diff(image, axis=1)
    
    # Compute the total variation loss
    tv_loss = np.sum(np.abs(pixel_diff_x)) + np.sum(np.abs(pixel_diff_y))
    return tv_loss

def get_variation_loss(img,x,y,a,b):
    image=img
    if type(img) is str:
        image = Image.open(img)
        image=image.convert('RGB')
    image=np.array(image)
    image= image[x:a+1,y:b+1]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tv_loss = total_variation_loss(gray_image)/gray_image.shape[0]/gray_image.shape[1] 
    return tv_loss

def check(image_path,x,y,a,b):
    image = Image.open(image_path)
    image=np.array(image)
    image= image[x:a+1,y:b+1]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image.sum()==0

def get_avgsaliency(saliency,img,x,y,a,b):
    image=img
    if type(img) is str:
        image = Image.open(img)
        image=image.convert('RGB')
    image=np.array(image)
    # plt.savefig('test'+str(x)+'.png')
    map=get_saliencyMap(saliency,image)
    map= map[x:a+1,y:b+1]
    return map.sum()/map.shape[0]/map.shape[1]

def get_saliencyMap(saliency,image):
    if type(image) is str:
        image = Image.open(image)
        image=image.convert('RGB')
    image=np.array(image)
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    # print(saliencyMap.sum()/saliencyMap.shape[0]/saliencyMap.shape[1])
    return saliencyMap

def calculate_clip_score(clip_score_fn, images, prompts):
    """
    Calculate CLIP score for images in batch
    Args:
        images: Can be either:
            - str: path to single image
            - list of str: paths to multiple images
            - numpy array: single image or batch of images
        prompts: text prompt or list of prompts
    Returns:
        list: CLIP scores for each image-prompt pair
    """
    if isinstance(images, str):
        # Single image path
        img = Image.open(images)
        img = img.convert('RGB')
        images = np.array(img)[np.newaxis, ...]
    elif isinstance(images, list) and isinstance(images[0], str):
        # List of image paths
        images = [np.array(Image.open(img_path).convert('RGB')) for img_path in images]
        images = np.stack(images)
    elif isinstance(images, np.ndarray) and images.ndim == 3:
        # Single image as numpy array
        images = images[np.newaxis, ...]
    
    # Convert to uint8 if not already
    if images.dtype != np.uint8:
        images = (images * 255).astype(np.uint8)
    
    # Convert to [N, C, H, W] format
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(device)
    
    # Ensure prompts is a list
    if isinstance(prompts, str):
        prompts = [prompts] * len(images_tensor)
    elif isinstance(prompts, list) and len(prompts) == 1:
        prompts = prompts * len(images_tensor)
    
    # 直接计算整个batch的CLIP分数
    scores = clip_score_fn(images_tensor, prompts).detach()
    return [round(float(scores.cpu()), 4)]  # 返回单个分数，因为clip_score内部会对batch取平均

def score_image(saliency, orig_path, edit_path, x, y, a, b):
    avgsaliency_orig = get_avgsaliency(saliency, orig_path, x, y, a, b)
    avgsaliency_edit = get_avgsaliency(saliency, edit_path, x, y, a, b)
    variation_loss_orig = get_variation_loss(orig_path, x, y, a, b)
    variation_loss_edit = get_variation_loss(edit_path, x, y, a, b)
    
    score = (avgsaliency_orig - avgsaliency_edit) + (variation_loss_orig - variation_loss_edit)
    return score

class min_heap:
    def __init__(self,size=10) -> None:
        self.heap=[]
        self.size=size

    def push(self,element):
        heapq.heappush(self.heap,element)
        if len(self.heap)>self.size:
            heapq.heappop(self.heap)

    def pop(self):
        return heapq.heappop(self.heap)

    def get(self):
        return self.heap[0]

    

def save_overlaid_image(img,heat_map,path):
    plt.figure()
    plt.imshow(img)
    plt.imshow(heat_map, cmap='jet', alpha=0.7)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_batch_entries(clip_score_fn, folder, entries, batch_size=30):
    """
    Process multiple entries in batches efficiently with CUDA
    """
    results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simple tqdm progress bar
    for i in tqdm(range(0, len(entries), batch_size)):
        batch_entries = entries[i:min(i+batch_size, len(entries))]
        batch_images_edit = []
        batch_images_orig = []
        batch_prompts = []
        batch_paths = []
        
        # Collect valid entries for the batch
        for entry in batch_entries:
            path = os.path.join(folder, entry)
            json_file = os.path.join(path, 'data.json')
            
            if not os.path.exists(json_file):
                continue
                
            with open(json_file) as f:
                data = json.load(f)
            prompt = data['prompt']
            
            edit_files = glob.glob(os.path.join(path, 'edited*.png'))
            orig_files = glob.glob(os.path.join(path, 'original*.png'))
            
            if not edit_files or not orig_files:
                continue
                
            edit_path = edit_files[0]
            orig_path = orig_files[0]
            
            if not (os.path.exists(edit_path) and os.path.exists(orig_path)):
                continue

            batch_images_edit.append(edit_path)
            batch_images_orig.append(orig_path)
            batch_prompts.append(prompt)
            batch_paths.append(path)
        
        # if not batch_images_edit:
        #     continue
            
        # Calculate scores for edit and original images in batch
        with torch.cuda.amp.autocast():
            clip_scores_edit = calculate_clip_score(clip_score_fn, batch_images_edit, batch_prompts)
            clip_scores_orig = calculate_clip_score(clip_score_fn, batch_images_orig, batch_prompts)
        
        # Store results
        for edit_score, orig_score, path in zip(clip_scores_edit, clip_scores_orig, batch_paths):
            clip_score_diff = edit_score - orig_score
            results.append((float(edit_score), float(orig_score), float(clip_score_diff), path))
            
    return results



    
    

# Example usage
# Load an image (for example, using OpenCV)
# image = cv2.imread('path_to_image.jpg', cv2.IMREAD_COLOR)

# Convert the image to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the total variation loss
# tv_loss = total_variation_loss(gray_image)
# print(f"Total Variation Loss: {tv_loss}")

# The above example is commented out as I cannot load images from external sources.
# You can uncomment and run this code with an actual image file.