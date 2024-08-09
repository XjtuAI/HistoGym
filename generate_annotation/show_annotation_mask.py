import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path as osp
import openslide
from pathlib import Path
import cv2 as cv2


slide_path = '/home/zhibo/data/minicamelyon16/tumor_001.tif'
truth_path = '/home/zhibo/data/minicamelyon16/mask/tumor_001_mask.tif'
wsi_level = 4

slide = openslide.open_slide(slide_path)
truth = openslide.open_slide(truth_path)

rgb_image = slide.read_region((0, 0), wsi_level, slide.level_dimensions[wsi_level])
rgb_mask = truth.read_region((0, 0), wsi_level, slide.level_dimensions[wsi_level])

grey = np.array(rgb_mask.convert('L'))
rgb_imagenew = np.array(rgb_image)

contours, _ = cv2.findContours(grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(rgb_imagenew, contours, -1, (0, 0, 255), 5)

# display original image with contours
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.imshow("output", rgb_imagenew)
cv2.waitKey(0)
