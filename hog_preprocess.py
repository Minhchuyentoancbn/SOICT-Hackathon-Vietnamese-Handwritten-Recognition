import cv2
import numpy as np
import os
from tqdm import tqdm

data_path = 'data/NAVER_OCR_private_test_update/'
output_dir = 'data/private_test/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

paths = [f for f in os.listdir(data_path)]

import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage.feature import hog
from skimage import exposure

def cut_(image: np.array, hog: np.array, thres: float):
    tmp_image = image.copy()
    H, W = tmp_image.shape[0], image.shape[1]
    
    crop_down = 0
    crop_up = H
    crop_left = 0
    crop_right = W
    
    thres_H = thres*H
    thres_W = thres*W
    
    for i in range(H):
        if hog[i, :].sum() > thres_H:
            crop_down = i-1
            break
    for i in reversed(range(H)):
        if hog[i, :].sum() > thres_H:
            crop_up = i+1
            break
    for j in range(W):
        if hog[:, j].sum() > thres_W:
            crop_left = j-1
            break
    for j in reversed(range(W)):
        if hog[:, j].sum() > thres_W:
            crop_right = j+1
            break
    return tmp_image[crop_down: crop_up+1, crop_left: crop_right+1, :]

def save_hog_(name):
    image = cv2.imread(os.path.join(data_path, name))  
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    fd, hog_image = hog(image, orientations=18, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    mean_grad = hog_image_rescaled.mean() 
    hog_image_rescaled = (hog_image_rescaled >= 0.7).astype(float)
    thres=mean_grad
    crop_img = cut_(image, hog_image_rescaled, mean_grad * 0.2)
    assert(crop_img.shape[-1]==3)
    plt.imsave(os.path.join(output_dir, f'./{name}'), crop_img)
    
for name in tqdm(paths, total=len(paths)):
    try:
        save_hog_(name)
    except:
        print("Exception:", name)
        continue